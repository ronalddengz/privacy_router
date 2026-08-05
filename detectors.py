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

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
except Exception:  # pragma: no cover - optional dependency in some environments
    AnalyzerEngine = None  # type: ignore[assignment]
    NlpEngineProvider = None  # type: ignore[assignment]

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
        "US_SSN": [
            r'\b\d{3}-\d{2}-\d{4}\b',
        ],
        "EMAIL_ADDRESS": [
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
        ],
        "PHONE_NUMBER": [
            r'\b(?:\+?1[-.\s]*)?(?:\(?\d{3}\)?[-.\s]*)\d{3}[-.\s]*\d{4}\b',
        ],
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
        if AnalyzerEngine is None or NlpEngineProvider is None:
            return None

        try:
            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
            )
            nlp_engine = provider.create_engine()
            return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        except Exception:
            return None

    def _detect_with_spacy(self, text: str) -> list[dict]:
        results = []
        if not hasattr(self, "_spacy_nlp"):
            self._spacy_nlp = self._load_spacy_model()

        if self._spacy_nlp is None:
            return results

        doc = self._spacy_nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "EMAIL", "PHONE_NUMBER", "NORP", "GPE", "LOC", "DATE", "TIME", "ORG"}:
                results.append({
                    "entity_type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "score": 0.7,
                    "detector": "spacy_ner",
                })
        return results

    def _load_spacy_model(self):
        try:
            import spacy
        except Exception:
            return None

        for model_name in ("en_core_web_sm", "en_core_web_lg"):
            try:
                return spacy.load(model_name)
            except Exception:
                continue
        return None
    
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
        if self.analyzer is not None:
            try:
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
                        calibrated_probability=None,
                    ))
            except Exception:
                pass
        
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

        # Lightweight spaCy fallback so PERSON/GPE/ORG entities are still captured
        for r in self._detect_with_spacy(text):
            category = self._categorize_entity(r["entity_type"], policy)
            evidence.append(PIIEvidence(
                entity_type=r["entity_type"],
                category=category,
                span_start=r["start"],
                span_end=r["end"],
                detector=r["detector"],
                detector_version=self.VERSION,
                raw_detector_score=r["score"],
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
        (r'\b(?:i\s*am|i\'m|aged?|age\s+is)\s*(\d{1,3})\b', 'exact'),
        (r'\b(\d{1,3})\s*(?:year[s]?\s*old|y/?o|yo)\b', 'exact'),
        (r'\bage[d]?\s*(\d{1,3})\b', 'exact'),
        (r'\b(\d{1,3})\s*(?:year[s]?\s*of\s*age)\b', 'exact'),
        (r'\bturn(?:ed)?\s*(\d{1,3})\b', 'exact'),
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

    SEX_PATTERNS = [
        (r'\b(?:sex|gender)\s*[:=]?\s*(male|female|man|woman|boy|girl|nonbinary|transgender)\b', 'contextual'),
        (r'\b(male|female|man|woman|boy|girl|nonbinary|transgender)\b', 'standalone'),
    ]

    MARITAL_STATUS_PATTERNS = [
        (r'\b(?:marital\s+status\s*[:=]?\s*)?(married|single|divorced|widowed|separated|never\s+married|never\s+been\s+married)\b', 'contextual'),
    ]

    CITIZENSHIP_PATTERNS = [
        (r'\b(?:us\s+citizen|american\s+citizen|citizen\s+of\s+the\s+united\s+states|naturalized\s+citizen)\b', 'citizen'),
        (r'\b(?:not\s+a\s+citizen|not\s+citizen|noncitizen|foreign[-\s]?born|immigrant)\b', 'noncitizen'),
        (r'\b(?:born\s+in\s+the\s+united\s+states|us[-\s]?born|born\s+in\s+the\s+us)\b', 'us_born'),
    ]

    RACE_ETHNICITY_PATTERNS = [
        (r'\b(?:laotian|hmong|cambodian|vietnamese|filipino|chinese|korean|japanese|asian|pacific\s+islander|native\s+hawaiian|white|black|african\s+american|american\s+indian|alaska\s+native|hispanic|latino|latina|middle\s+eastern|arab)\b', 'race_ethnicity'),
    ]

    STATE_NAME_TO_ABBR = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
        'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'district of columbia': 'DC',
        'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL',
        'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA',
        'maine': 'ME', 'maryland': 'MD', 'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN',
        'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
        'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
        'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR',
        'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD',
        'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA',
        'washington': 'WA', 'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
    }

    CITIZENSHIP_NORMALIZATION = {
        'citizen': 'citizen',
        'us_citizen': 'citizen',
        'american_citizen': 'citizen',
        'naturalized': 'naturalized',
        'naturalized_citizen': 'naturalized',
        'noncitizen': 'noncitizen',
        'not_a_citizen': 'noncitizen',
        'not_citizen': 'noncitizen',
        'immigrant': 'noncitizen',
        'foreign_born': 'noncitizen',
        'us_born': 'us_born',
        'born_in_the_united_states': 'us_born',
        'born_in_the_us': 'us_born',
    }
    
    def __init__(self):
        self.nlp = self._load_spacy_model()

    def _load_spacy_model(self):
        try:
            import spacy
        except Exception:
            return None

        for model_name in ("en_core_web_sm", "en_core_web_lg"):
            try:
                return spacy.load(model_name)
            except Exception:
                continue
        return None

    def _append_qi(
        self,
        evidence: list[QIEvidence],
        qi_type: str,
        normalized_value: str,
        granularity: str,
        detector: str,
        confidence: float,
    ) -> None:
        evidence.append(QIEvidence(
            qi_type=qi_type,
            normalized_value=normalized_value,
            granularity=granularity,
            detector=detector,
            detector_version=self.VERSION,
            extraction_confidence=confidence,
        ))
    
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
                        self._append_qi(
                            evidence,
                            qi_type="age_5yr",
                            normalized_value=f"{bucket_start}-{bucket_end}",
                            granularity="5yr_bucket",
                            detector="pattern",
                            confidence=0.9,
                        )
                except ValueError:
                    pass

        # 1b. Sex/gender detection
        for pattern, granularity in self.SEX_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                value = match.group(1).lower()
                normalized = {
                    'man': 'male',
                    'boy': 'male',
                    'male': 'male',
                    'woman': 'female',
                    'girl': 'female',
                    'female': 'female',
                    'nonbinary': 'nonbinary',
                    'transgender': 'transgender',
                }.get(value, value)
                self._append_qi(
                    evidence,
                    qi_type="sex",
                    normalized_value=normalized,
                    granularity="binary_or_gender_identity",
                    detector="pattern",
                    confidence=0.85,
                )

        # 1c. Marital status detection
        for pattern, granularity in self.MARITAL_STATUS_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                raw_value = match.group(1).lower()
                normalized = raw_value.replace(" ", "_").replace("been_", "")
                if normalized == "never_been_married":
                    normalized = "never_married"
                elif normalized == "single":
                    normalized = "single"
                self._append_qi(
                    evidence,
                    qi_type="marital_status",
                    normalized_value=normalized,
                    granularity="marital_status",
                    detector="pattern",
                    confidence=0.82,
                )

        # 1d. Citizenship detection
        for pattern, normalized in self.CITIZENSHIP_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                normalized = self.CITIZENSHIP_NORMALIZATION.get(normalized, normalized)
                self._append_qi(
                    evidence,
                    qi_type="citizenship",
                    normalized_value=normalized,
                    granularity="citizenship_status",
                    detector="pattern",
                    confidence=0.8,
                )

        # 1e. Race/ethnicity detection
        for pattern, qi_type in self.RACE_ETHNICITY_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                normalized = match.group(0).lower().replace(" ", "_")
                self._append_qi(
                    evidence,
                    qi_type=qi_type,
                    normalized_value=normalized,
                    granularity="census_race_ethnicity",
                    detector="pattern",
                    confidence=0.78,
                )
        
        # 2. Location detection via spaCy
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "GPE":  # Geo-political entity
                    state_name = ent.text.strip().lower()
                    if state_name in self.STATE_NAME_TO_ABBR:
                        self._append_qi(
                            evidence,
                            qi_type="state",
                            normalized_value=self.STATE_NAME_TO_ABBR[state_name],
                            granularity="state",
                            detector="spacy_ner",
                            confidence=0.7,
                        )
                    elif state_name in {'oregon', 'oregon state'}:
                        self._append_qi(
                            evidence,
                            qi_type="state",
                            normalized_value='OR',
                            granularity="state",
                            detector="spacy_ner",
                            confidence=0.7,
                        )
                    else:
                        self._append_qi(
                            evidence,
                            qi_type="location",
                            normalized_value=ent.text.lower(),
                            granularity="unknown",
                            detector="spacy_ner",
                            confidence=0.7,
                        )

        # 2b. Direct state-name matching for narrative benchmark text
        for state_name, state_abbr in self.STATE_NAME_TO_ABBR.items():
            pattern = rf'\b{re.escape(state_name)}\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="state",
                    normalized_value=state_abbr,
                    granularity="state",
                    detector="gazetteer",
                    confidence=0.88,
                )
        if re.search(r'\boregon\b', text_lower):
            self._append_qi(
                evidence,
                qi_type="state",
                normalized_value='OR',
                granularity="state",
                detector="gazetteer",
                confidence=0.88,
            )
        
        # 3. Rare disease detection
        for term in self.RARE_DISEASE_TERMS:
            if term in text_lower:
                self._append_qi(
                    evidence,
                    qi_type="condition",
                    normalized_value=term,
                    granularity="specific_disease",
                    detector="gazetteer",
                    confidence=0.85,
                )
        
        # 4. Rare occupation detection
        for term in self.RARE_OCCUPATION_TERMS:
            if term in text_lower:
                self._append_qi(
                    evidence,
                    qi_type="occupation",
                    normalized_value=term,
                    granularity="specific_occupation",
                    detector="gazetteer",
                    confidence=0.85,
                )
        
        # 5. Date detection (temporal QIs)
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="date",
                    normalized_value=match.group(0),
                    granularity="exact_date",
                    detector="pattern",
                    confidence=0.85,
                )
        
        # Multiple rules commonly describe the same fact (for example the
        # standalone and labelled sex patterns, or a state found by both
        # spaCy and the gazetteer). Keep one QI per normalized value so the
        # risk estimator does not count detector duplicates as additional
        # identifying attributes.
        deduplicated: dict[tuple[str, Optional[str]], QIEvidence] = {}
        for qi in evidence:
            key = (qi.qi_type, qi.normalized_value)
            existing = deduplicated.get(key)
            if existing is None:
                deduplicated[key] = qi
                continue
            if qi.extraction_confidence > existing.extraction_confidence:
                existing.extraction_confidence = qi.extraction_confidence
            if qi.detector and qi.detector != existing.detector:
                if qi.detector not in existing.detector_agreement:
                    existing.detector_agreement.append(qi.detector)

        return list(deduplicated.values())
    
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
