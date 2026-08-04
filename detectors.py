"""
detectors.py - Fast deterministic PII and QI detection

Layered detection approach:
1. Structured patterns (regex, rules)
2. Terminology/gazetteers  
3. Fast NER (spaCy/Presidio)
4. Ensemble combination with conservative masking
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NoOpNlpEngine

from policy import (
    PIIEvidence, QIEvidence, EntityCategory, HarmCategory, PolicyProfile
)


class PIIDetector:
    """
    Fast PII detection using Presidio + custom patterns.
    
    Returns structured evidence, not scores.
    Direct identifiers are flagged for mandatory masking regardless of confidence.
    """
    
    VERSION = "1.0.0"
    
    # Additional patterns not in Presidio
    CUSTOM_PATTERNS = {
        "MEDICAL_RECORD_NUMBER": [
            r'\bMRN[:\s#]*\d{6,12}\b',
            r'\bMR[:\s#]*\d{6,12}\b',
            r'\bPatient\s*ID[:\s#]*\d{6,12}\b',
        ],
        "HEALTH_PLAN_ID": [
            r'\b[A-Z]{3}\d{9,12}\b',  # Common insurance ID format
        ],
    }
    
    def __init__(self, score_threshold: float = 0.3):
        self.score_threshold = score_threshold
        self.analyzer = self._create_analyzer()
    
    def _create_analyzer(self) -> AnalyzerEngine:
        nlp_engine = NoOpNlpEngine(
            models=[{"lang_code": "en", "model_name": "no_op"}]
        )
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    
    def _categorize_entity(self, entity_type: str, policy: PolicyProfile) -> EntityCategory:
        """Categorize entity type for policy routing."""
        if entity_type in policy.direct_identifier_types:
            return EntityCategory.DIRECT_IDENTIFIER
        elif entity_type in {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}:
            return EntityCategory.STRONG_IDENTIFIER
        elif entity_type in {"LOCATION", "DATE_TIME", "ORGANIZATION"}:
            return EntityCategory.CONTEXTUAL
        else:
            return EntityCategory.QUASI_IDENTIFIER
    
    def _detect_custom_patterns(self, text: str) -> list[dict]:
        """Apply custom regex patterns."""
        results = []
        for entity_type, patterns in self.CUSTOM_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    results.append({
                        'entity_type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'score': 0.9,  # High confidence for regex matches
                        'detector': 'custom_regex'
                    })
        return results
    
    def detect(self, text: str, policy: PolicyProfile) -> list[PIIEvidence]:
        """
        Detect PII/PHI entities in text.
        
        Returns structured evidence for each detection.
        Does NOT compute aggregate scores - that's a policy decision.
        """
        evidence = []
        
        # Presidio detection
        presidio_results = self.analyzer.analyze(
            text=text,
            language="en",
            score_threshold=self.score_threshold
        )
        
        for r in presidio_results:
            category = self._categorize_entity(r.entity_type, policy)
            evidence.append(PIIEvidence(
                entity_type=r.entity_type,
                category=category,
                span_start=r.start,
                span_end=r.end,
                detector="presidio",
                detector_version=self.VERSION,
                raw_detector_score=r.score,
                calibrated_probability=None,  # Would need calibration data
            ))
        
        # Custom pattern detection
        custom_results = self._detect_custom_patterns(text)
        for r in custom_results:
            category = self._categorize_entity(r['entity_type'], policy)
            evidence.append(PIIEvidence(
                entity_type=r['entity_type'],
                category=category,
                span_start=r['start'],
                span_end=r['end'],
                detector=r['detector'],
                detector_version=self.VERSION,
                raw_detector_score=r['score'],
                calibrated_probability=None,
            ))
        
        # Merge overlapping detections
        evidence = self._merge_overlapping(evidence)
        
        return evidence
    
    def _merge_overlapping(self, evidence: list[PIIEvidence]) -> list[PIIEvidence]:
        """Merge overlapping spans, keeping highest-priority category."""
        if not evidence:
            return []
        
        # Sort by start position
        sorted_evidence = sorted(evidence, key=lambda e: (e.span_start, e.span_end))
        merged = [sorted_evidence[0]]
        
        priority = {
            EntityCategory.DIRECT_IDENTIFIER: 0,
            EntityCategory.STRONG_IDENTIFIER: 1,
            EntityCategory.QUASI_IDENTIFIER: 2,
            EntityCategory.CONTEXTUAL: 3,
        }
        
        for current in sorted_evidence[1:]:
            last = merged[-1]
            
            # Check overlap
            if current.span_start < last.span_end:
                # Merge: extend span, keep higher priority
                if priority[current.category] < priority[last.category]:
                    merged[-1] = PIIEvidence(
                        entity_type=current.entity_type,
                        category=current.category,
                        span_start=min(last.span_start, current.span_start),
                        span_end=max(last.span_end, current.span_end),
                        detector=current.detector,
                        detector_version=current.detector_version,
                        raw_detector_score=max(last.raw_detector_score, current.raw_detector_score),
                        calibrated_probability=None,
                        confirmed_by=[last.detector] if last.detector != current.detector else [],
                        merged_from_overlapping=True,
                    )
                else:
                    # Keep last, but note confirmation
                    last.confirmed_by.append(current.detector)
                    merged[-1].span_end = max(last.span_end, current.span_end)
                    merged[-1].merged_from_overlapping = True
            else:
                merged.append(current)
        
        return merged


class QIDetector:
    """
    Quasi-identifier detection using layered approach.
    
    Layers:
    1. Structured patterns (ages, dates, ZIP codes)
    2. Terminology matching (conditions, occupations, locations)
    3. spaCy NER for residual entities
    
    No LLM at this stage - that's for contextual analysis only.
    """
    
    VERSION = "1.0.0"
    
    # Age patterns
    AGE_PATTERNS = [
        (r'\b(\d{1,3})\s*(?:year[s]?\s*old|y/?o|yo)\b', 'exact'),
        (r'\bage[d]?\s*(\d{1,3})\b', 'exact'),
        (r'\b(\d{1,3})\s*(?:year[s]?\s*of\s*age)\b', 'exact'),
    ]
    
    # ZIP code patterns
    ZIP_PATTERNS = [
        (r'\b(\d{5})-\d{4}\b', 'zip9'),  # ZIP+4
        (r'\bzip\s*(?:code)?\s*(\d{5})\b', 'zip5'),
        (r'\b(\d{5})\b', 'zip5'),  # Plain 5-digit (needs context)
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
    ]
    
    # Common rare disease terms (would be loaded from ontology in production)
    RARE_DISEASE_TERMS = {
        'ehlers-danlos', 'ehlers danlos', 'eds', 'hypermobility syndrome',
        'huntington', 'huntingtons', "huntington's",
        'marfan', 'cystic fibrosis', 'cf',
        'als', 'amyotrophic lateral sclerosis',
        'sickle cell', 'hemophilia', 'tay-sachs',
        'duchenne', 'muscular dystrophy',
    }
    
    # Occupation terms (subset - would load from BLS SOC codes)
    RARE_OCCUPATION_TERMS = {
        'zoologist', 'astronomer', 'epidemiologist',
        'coroner', 'medical examiner', 'air traffic controller',
        'nuclear engineer', 'geologist', 'archaeologist',
    }
    
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_lg")
    
    def detect(self, text: str) -> list[QIEvidence]:
        """Detect quasi-identifiers in text."""
        evidence = []
        text_lower = text.lower()
        
        # 1. Age detection
        for pattern, granularity in self.AGE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                age_str = match.group(1)
                try:
                    age = int(age_str)
                    if 0 <= age <= 120:  # Reasonable age range
                        # Normalize to 5-year bucket
                        bucket_start = (age // 5) * 5
                        bucket_end = bucket_start + 4
                        evidence.append(QIEvidence(
                            qi_type="age_5yr",
                            normalized_value=f"{bucket_start}-{bucket_end}",
                            granularity="5yr_bucket",
                            detector="pattern",
                            detector_version=self.VERSION,
                            extraction_confidence=0.9,
                        ))
                except ValueError:
                    pass
        
        # 2. Location detection via spaCy
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Geo-political entity
                evidence.append(QIEvidence(
                    qi_type="location",
                    normalized_value=ent.text.lower(),
                    granularity="unknown",  # Would need geocoding to determine
                    detector="spacy_ner",
                    detector_version=self.VERSION,
                    extraction_confidence=0.7,
                ))
        
        # 3. Rare disease detection
        for term in self.RARE_DISEASE_TERMS:
            if term in text_lower:
                evidence.append(QIEvidence(
                    qi_type="condition",
                    normalized_value=term,
                    granularity="specific_disease",
                    detector="gazetteer",
                    detector_version=self.VERSION,
                    extraction_confidence=0.85,
                    # This would trigger high-harm category
                ))
        
        # 4. Rare occupation detection
        for term in self.RARE_OCCUPATION_TERMS:
            if term in text_lower:
                evidence.append(QIEvidence(
                    qi_type="occupation",
                    normalized_value=term,
                    granularity="specific_occupation",
                    detector="gazetteer",
                    detector_version=self.VERSION,
                    extraction_confidence=0.85,
                ))
        
        # 5. Date detection (temporal QIs)
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                evidence.append(QIEvidence(
                    qi_type="date",
                    normalized_value=match.group(0),
                    granularity="exact_date",
                    detector="pattern",
                    detector_version=self.VERSION,
                    extraction_confidence=0.85,
                ))
        
        return evidence
    
    def check_assertion_status(self, text: str, qi: QIEvidence) -> QIEvidence:
        """
        Check if QI is negated, hypothetical, or refers to someone other than patient.
        
        Important for medical QIs - "mother has Huntington's" is different from
        "patient has Huntington's".
        """
        # This would use a proper clinical NLP model in production
        # Simple heuristic here
        text_lower = text.lower()
        
        # Check for negation (very simplified)
        negation_patterns = [
            r'no\s+(?:history\s+of\s+)?' + re.escape(qi.normalized_value or ''),
            r'denies\s+' + re.escape(qi.normalized_value or ''),
            r'negative\s+for\s+' + re.escape(qi.normalized_value or ''),
        ]
        for pattern in negation_patterns:
            if re.search(pattern, text_lower):
                qi.assertion_status = "negated"
                break
        
        # Check for family history
        family_patterns = [
            r'(?:mother|father|sister|brother|parent|sibling|family)\s+(?:has|had|with)\s+',
            r'family\s+history\s+of\s+',
        ]
        for pattern in family_patterns:
            if re.search(pattern + re.escape(qi.normalized_value or ''), text_lower):
                qi.experiencer = "family"
                break
        
        return qi