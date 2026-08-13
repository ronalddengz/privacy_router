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
        "DATE": [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        ],
        "TIME": [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b',
            r'\b\d{1,2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b',
            r'\b(?:at\s+)?\d{1,2}:\d{2}(?::\d{2})?(?!\d)\b',
        ],
        "DATE_TIME": [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b',
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
        if entity_type in policy.direct_identifier_types:
            return EntityCategory.DIRECT_IDENTIFIER
        elif entity_type in {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}:
            return EntityCategory.STRONG_IDENTIFIER
        elif entity_type in {"LOCATION", "GPE", "LOC", "ORGANIZATION", "ORG", "FAC", "DATE_TIME"}:
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
        
        # detectors.py -> PIIDetector.detect()
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
            except Exception as e:
                print(f"[Warning] Presidio detection error: {e}")
        
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
                    if hasattr(last, 'confirmed_by') and last.confirmed_by is not None:
                        last.confirmed_by.append(current.detector)
                    merged[-1].span_end = max(last.span_end, current.span_end)
                    merged[-1].merged_from_overlapping = True
            else:
                merged.append(current)
        
        return merged


class QIDetector:
    """
    Detects quasi-identifiers in text with proper normalization to match
    frequency table canonical values for accurate k-anonymity estimation.
    """
    
    VERSION = "1.1.0"
    
    # -------------------------------------------------------------------------
    # SEX PATTERNS - maps to: "male", "female"
    # -------------------------------------------------------------------------
    SEX_PATTERNS = [
        (r'\b(?:sex|gender)\s*[:=]?\s*(male|female|man|woman|boy|girl)\b', 'contextual'),
        (r'\b(?:i\s+am\s+)?(male|female|a\s+man|a\s+woman)\b', 'statement'),
        (r'\b(male|female)\b', 'standalone'),
    ]
    
    # O(1) lookup for sex normalization
    SEX_NORMALIZATION = {
        'male': 'male',
        'man': 'male',
        'boy': 'male',
        'a man': 'male',
        'female': 'female',
        'woman': 'female',
        'girl': 'female',
        'a woman': 'female',
    }
    
    # -------------------------------------------------------------------------
    # MARITAL STATUS PATTERNS - maps to: "married", "single", "divorced", 
    #                                     "widowed", "separated"
    # -------------------------------------------------------------------------
    MARITAL_STATUS_PATTERNS = [
        (r"\b(?:marital\s+status\s*[:=]?\s*)?(married|single|divorced|widowed|separated|never\s+married|never\s+been\s+married|unmarried|i'?m\s+married|i'?m\s+single|i'?m\s+divorced|i'?m\s+widowed|i'?m\s+separated)\b", 'contextual'),
        (r"\bmy\s+(?:spouse|wife|husband)\b", 'married_indicator'),
    ]
    
    # O(1) lookup for marital status normalization
    MARITAL_STATUS_NORMALIZATION = {
        'married': 'married',
        "i'm married": 'married',
        'im married': 'married',
        'single': 'single',
        "i'm single": 'single',
        'im single': 'single',
        'never married': 'single',
        'never been married': 'single',
        'unmarried': 'single',
        'divorced': 'divorced',
        "i'm divorced": 'divorced',
        'im divorced': 'divorced',
        'widowed': 'widowed',
        "i'm widowed": 'widowed',
        'im widowed': 'widowed',
        'separated': 'separated',
        "i'm separated": 'separated',
        'im separated': 'separated',
    }
    
    # -------------------------------------------------------------------------
    # CITIZENSHIP PATTERNS - maps to: "citizen", "naturalized", "noncitizen", 
    #                                  "us_born"
    # -------------------------------------------------------------------------
    CITIZENSHIP_PATTERNS = [
        # Naturalized citizen patterns (must come before general citizen)
        (r'\b(?:u\.?s\.?\s+citizen\s+by\s+naturalization|citizen\s+by\s+naturalization|naturalized\s+citizen|naturalized\s+u\.?s\.?\s+citizen|became\s+a\s+citizen|citizenship\s+through\s+naturalization)\b', 'naturalized'),
        # US-born patterns (including territories)
        (r'\b(?:born\s+in\s+(?:the\s+)?(?:united\s+states|u\.?s\.?|usa|puerto\s+rico|guam|u\.?s\.?\s+virgin\s+islands|american\s+samoa)|us[-\s]?born|native[-\s]?born\s+citizen)\b', 'us_born'),
        # General citizen patterns
        (r'\b(?:u\.?s\.?\s+citizen|american\s+citizen|citizen\s+of\s+(?:the\s+)?united\s+states|i\s+am\s+a\s+citizen)\b', 'citizen'),
        # Non-citizen patterns
        (r'\b(?:not\s+a\s+citizen|not\s+a\s+u\.?s\.?\s+citizen|i\s+am\s+not\s+a\s+citizen|noncitizen|non[-\s]?citizen|foreign[-\s]?national|foreign[-\s]?born(?!\s+citizen))\b', 'noncitizen'),
        # Explicit "citizenship status is not a citizen"
        (r'\bcitizenship\s+status\s+is\s+not\s+a\s+citizen\b', 'noncitizen'),
    ]
    
    # -------------------------------------------------------------------------
    # RACE/ETHNICITY PATTERNS - maps to: "white", "black", "asian", "hispanic",
    #                                     "native_american", "pacific_islander",
    #                                     "laotian", "hmong"
    # -------------------------------------------------------------------------
    # Detection pattern captures all variations
    RACE_ETHNICITY_PATTERN = re.compile(
        r'\b(?:my\s+race\s+is\s+)?('
        r'laotian|hmong|cambodian|vietnamese|filipino|filipina|chinese|korean|japanese|'
        r'asian\s+indian|asian|pacific\s+islander|native\s+hawaiian|samoan|tongan|'
        r'white|caucasian|black|african\s+american|african[-\s]?american|'
        r'american\s+indian|alaska\s+native|native\s+american|indigenous|'
        r'hispanic|latino|latina|latinx|mexican|puerto\s+rican|cuban|'
        r'middle\s+eastern|arab|two\s+or\s+more\s+races|multiracial|mixed\s+race'
        r')\b',
        re.IGNORECASE
    )
    
    # O(1) lookup for race/ethnicity normalization to frequency table values
    RACE_ETHNICITY_NORMALIZATION = {
        # White
        'white': 'white',
        'caucasian': 'white',
        # Black
        'black': 'black',
        'african american': 'black',
        'african-american': 'black',
        # Asian (general and specific - most map to "asian")
        'asian': 'asian',
        'asian indian': 'asian',
        'chinese': 'asian',
        'korean': 'asian',
        'japanese': 'asian',
        'vietnamese': 'asian',
        'filipino': 'asian',
        'filipina': 'asian',
        'cambodian': 'asian',
        # Specific Asian ethnicities with their own frequency table entries
        'laotian': 'laotian',
        'hmong': 'hmong',
        # Hispanic/Latino
        'hispanic': 'hispanic',
        'latino': 'hispanic',
        'latina': 'hispanic',
        'latinx': 'hispanic',
        'mexican': 'hispanic',
        'puerto rican': 'hispanic',
        'cuban': 'hispanic',
        # Native American
        'native american': 'native_american',
        'american indian': 'native_american',
        'alaska native': 'native_american',
        'indigenous': 'native_american',
        # Pacific Islander
        'pacific islander': 'pacific_islander',
        'native hawaiian': 'pacific_islander',
        'samoan': 'pacific_islander',
        'tongan': 'pacific_islander',
        # Multi-racial (no direct frequency table match, mark as unseen)
        'two or more races': None,  # Will be flagged as unseen
        'multiracial': None,
        'mixed race': None,
        # Middle Eastern (no direct match)
        'middle eastern': None,
        'arab': None,
    }
    
    # -------------------------------------------------------------------------
    # STATE DETECTION - maps to 2-letter abbreviations: "AL", "AK", etc.
    # -------------------------------------------------------------------------
    STATE_NAME_TO_ABBR = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
        'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
        'district of columbia': 'DC', 'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI',
        'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
        'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME',
        'maryland': 'MD', 'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN',
        'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE',
        'nevada': 'NV', 'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM',
        'new york': 'NY', 'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH',
        'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI',
        'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX',
        'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
        'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
    }
    
    # Valid 2-letter state codes for direct matching
    VALID_STATE_CODES = set(STATE_NAME_TO_ABBR.values())
    
    # Pattern to match 2-letter state codes in context
    STATE_CODE_PATTERN = re.compile(
        r'\b(?:in|from|living\s+in|reside\s+in|state\s+of|,)\s*([A-Z]{2})\b'
    )
    
    # -------------------------------------------------------------------------
    # OCCUPATION PATTERNS - maps to: "healthcare", "education", "zoologist"
    # -------------------------------------------------------------------------
    # The frequency table only has these three occupation categories
    OCCUPATION_KEYWORDS = {
        # Healthcare
        'doctor': 'healthcare',
        'physician': 'healthcare',
        'nurse': 'healthcare',
        'medical': 'healthcare',
        'hospital': 'healthcare',
        'clinic': 'healthcare',
        'healthcare': 'healthcare',
        'health care': 'healthcare',
        'pharmacist': 'healthcare',
        'therapist': 'healthcare',
        'surgeon': 'healthcare',
        'dentist': 'healthcare',
        'veterinarian': 'healthcare',
        # Education
        'teacher': 'education',
        'professor': 'education',
        'educator': 'education',
        'instructor': 'education',
        'school': 'education',
        'university': 'education',
        'college': 'education',
        'principal': 'education',
        'tutor': 'education',
        # Rare occupation (explicit)
        'zoologist': 'zoologist',
    }
    
    # Rare occupations that trigger high-harm checks
    RARE_OCCUPATION_TERMS = {'zoologist', 'astronomer', 'epidemiologist'}
    
    # -------------------------------------------------------------------------
    # RARE DISEASE TERMS (unchanged from original)
    # -------------------------------------------------------------------------
    RARE_DISEASE_TERMS = {
        'ehlers-danlos', 'ehlers danlos', 'huntington', 'huntingtons',
        'marfan', 'cystic fibrosis',
    }
    
    # -------------------------------------------------------------------------
    # AGE AND DATE PATTERNS (unchanged per constraints)
    # -------------------------------------------------------------------------
    AGE_PATTERNS = [
        (r'\b(?:age|aged)\s*[:=]?\s*(\d{1,3})\b', 'contextual'),
        (r'\b(\d{1,3})\s*(?:years?\s+old|y/?o|yo)\b', 'contextual'),
        (r"\bI(?:'m|am)\s+(\d{1,3})\b", 'statement'),
    ]
    
    DATE_PATTERNS = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 01/15/2024, 1-15-24
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # January 15, 2024
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',  # 15 January 2024
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',  # Jan 15, 2024
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # ISO format: 2024-01-15
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',  # January 2024 (month-year)
    ]

    TIME_PATTERNS = [
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?\b',  # 10:30 AM, 10:30:45 PM
        r'\b\d{1,2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b',  # 10 AM, 3pm
        r'\b(?:at\s+)?\d{1,2}:\d{2}(?::\d{2})?\b',  # at 10:30, 14:30:00
        r'\b(?:noon|midnight)\b',  # noon, midnight
        r'\b\d{1,2}\s*(?:o\'?clock)\b',  # 3 o'clock
    ]

    DATETIME_PATTERNS = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',  # 01/15/2024 10:30 AM
        r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b',  # ISO 8601: 2024-01-15T10:30:00Z
    ]

    # -------------------------------------------------------------------------
    # EDUCATIONAL ATTAINMENT PATTERNS - maps to SCHL codes
    # -------------------------------------------------------------------------
    EDUCATION_PATTERNS = [
        (r'\b(?:completed?|have|earned?|holds?|with)\s+(?:a\s+)?(high school|hs)\s*(?:diploma|degree)?\b', 'high_school_diploma'),
        (r'\b(?:ged|g\.e\.d\.)\b', 'ged'),
        (r'\bsome\s+college\b', 'some_college_1yr_plus'),
        (r'\b(?:associate[\'s]?s?|aa|as)\s*(?:degree)?\b', 'associates_degree'),
        (r'\b(?:bachelor[\'s]?s?|ba|bs|b\.a\.|b\.s\.)\s*(?:degree)?\b', 'bachelors_degree'),
        (r'\b(?:master[\'s]?s?|ma|ms|mba|m\.a\.|m\.s\.)\s*(?:degree)?\b', 'masters_degree'),
        (r'\b(?:doctorate|doctoral|ph\.?d\.?|phd)\b', 'doctorate_degree'),
        (r'\b(?:md|m\.d\.|medical\s+degree|doctor\s+of\s+medicine)\b', 'professional_degree'),
        (r'\b(?:jd|j\.d\.|law\s+degree|juris\s+doctor)\b', 'professional_degree'),
        (r'\bdropped?\s+out\b', 'grade_12_no_diploma'),
        (r'\bno\s+(?:formal\s+)?education\b', 'no_schooling'),
    ]
    
    # -------------------------------------------------------------------------
    # EMPLOYMENT STATUS PATTERNS - maps to ESR codes
    # -------------------------------------------------------------------------
    EMPLOYMENT_STATUS_PATTERNS = [
        (r'\b(?:currently\s+)?(?:employed|working|at\s+work)\b', 'employed_at_work'),
        (r'\b(?:unemployed|out\s+of\s+work|jobless|looking\s+for\s+(?:a\s+)?(?:job|work))\b', 'unemployed'),
        (r'\b(?:retired|not\s+working)\b', 'not_in_labor_force'),
        (r'\b(?:in\s+the\s+)?(?:military|armed\s+forces|army|navy|air\s+force|marines?|coast\s+guard)\b', 'armed_forces_at_work'),
        (r'\b(?:serving|enlisted|active\s+duty)\b', 'armed_forces_at_work'),
        (r'\b(?:stay[\s-]?at[\s-]?home|homemaker|housewife|househusband)\b', 'not_in_labor_force'),
        (r'\b(?:student|in\s+school|attending\s+(?:college|university))\b', 'not_in_labor_force'),
        (r'\b(?:disabled|on\s+disability)\b', 'not_in_labor_force'),
    ]
    
    # -------------------------------------------------------------------------
    # DETAILED OCCUPATION PATTERNS - maps to SOC codes
    # -------------------------------------------------------------------------
    OCCUPATION_KEYWORDS = {
        # Healthcare (SOC 29, 31)
        'doctor': 'healthcare_practitioners',
        'physician': 'healthcare_practitioners',
        'nurse': 'healthcare_practitioners',
        'registered nurse': 'registered_nurse',
        'medical': 'healthcare_practitioners',
        'hospital': 'healthcare_practitioners',
        'healthcare': 'healthcare_practitioners',
        'pharmacist': 'healthcare_practitioners',
        'therapist': 'healthcare_practitioners',
        'surgeon': 'healthcare_practitioners',
        'dentist': 'healthcare_practitioners',
        
        # Education (SOC 25)
        'teacher': 'education_training_library',
        'professor': 'postsecondary_teacher',
        'educator': 'education_training_library',
        'instructor': 'education_training_library',
        'school': 'education_training_library',
        
        # Business/HR (SOC 13)
        'human resources': 'human_resources_specialist',
        'hr specialist': 'human_resources_specialist',
        'hr worker': 'human_resources_specialist',
        'accountant': 'accountant_auditor',
        'auditor': 'accountant_auditor',
        
        # Science (SOC 19)
        'biological technician': 'biological_technician',
        'biologist': 'life_physical_social_science',
        'environmental scientist': 'environmental_scientist',
        'zoologist': 'zoologist',
        'astronomer': 'astronomer',
        'epidemiologist': 'epidemiologist',
        'scientist': 'life_physical_social_science',
        
        # Technology (SOC 15)
        'software developer': 'software_developer',
        'programmer': 'software_developer',
        'computer': 'computer_mathematical',
        'systems analyst': 'computer_systems_analyst',
        
        # Legal (SOC 23)
        'lawyer': 'lawyer',
        'attorney': 'lawyer',
        'legal': 'legal',
        
        # Protective Service (SOC 33)
        'police': 'police_officer',
        'firefighter': 'firefighter',
        'fire fighter': 'firefighter',
        
        # Sales (SOC 41)
        'salesperson': 'sales',
        'retail': 'retail_salesperson',
        
        # Transportation (SOC 53)
        'truck driver': 'truck_driver',
        'trucker': 'truck_driver',
        
        # Construction (SOC 47)
        'construction': 'construction_extraction',
        'laborer': 'construction_laborer',
        
        # Management (SOC 11)
        'manager': 'management',
        'executive': 'management',
        'ceo': 'management',
        'director': 'management',
    }
    
    # Rare occupations that trigger high-harm checks
    RARE_OCCUPATION_TERMS = {'zoologist', 'astronomer', 'epidemiologist'}
    
    # -------------------------------------------------------------------------
    # DETAILED RACE/ETHNICITY NORMALIZATION - maps to RAC2P categories
    # -------------------------------------------------------------------------
    RACE_ETHNICITY_NORMALIZATION = {
        # White
        'white': 'white',
        'caucasian': 'white',
        # Black
        'black': 'black',
        'african american': 'black',
        'african-american': 'black',
        # Asian - detailed
        'asian': 'asian',
        'asian indian': 'asian_indian',
        'indian': 'asian_indian',  # Contextual - could be Native American
        'chinese': 'chinese',
        'korean': 'korean',
        'japanese': 'japanese',
        'vietnamese': 'vietnamese',
        'filipino': 'filipino',
        'filipina': 'filipino',
        'cambodian': 'cambodian',
        'laotian': 'laotian',
        'hmong': 'hmong',
        'thai': 'thai',
        'pakistani': 'pakistani',
        'bangladeshi': 'bangladeshi',
        # Pacific Islander - detailed
        'pacific islander': 'pacific_islander',
        'native hawaiian': 'native_hawaiian',
        'hawaiian': 'native_hawaiian',
        'samoan': 'samoan',
        'tongan': 'tongan',
        'guamanian': 'guamanian_chamorro',
        'chamorro': 'guamanian_chamorro',
        # Hispanic/Latino - detailed
        'hispanic': 'hispanic',
        'latino': 'hispanic',
        'latina': 'hispanic',
        'latinx': 'hispanic',
        'mexican': 'mexican',
        'puerto rican': 'puerto_rican',
        'cuban': 'cuban',
        'dominican': 'dominican',
        # Native American - detailed
        'native american': 'native_american',
        'american indian': 'native_american',
        'alaska native': 'alaska_native',
        'indigenous': 'native_american',
        'cherokee': 'cherokee',
        'navajo': 'navajo',
        'sioux': 'sioux',
        'choctaw': 'choctaw',
        'chippewa': 'chippewa',
        'apache': 'apache',
        # Multi-racial
        'two or more races': 'two_or_more_races',
        'multiracial': 'two_or_more_races',
        'mixed race': 'two_or_more_races',
        'biracial': 'two_or_more_races',
    }

    def redact_qi_text(self, text: str, qi_evidences: list, mask_format="category") -> str:
        """
        Redacts detected quasi-identifiers from the text using detected QIEvidence offsets.
        
        :param text: Original raw text string.
        :param qi_evidences: List of QIEvidence objects containing start_char, end_char, entity_type/category.
        :param mask_format: "category" for [LOCATION] or "generic" for [REDACTED_QI].
        """
        if not qi_evidences:
            return text

        # 1. Sort evidence spans in REVERSE order by start_char / span_start
        # This prevents character shift issues during string replacement.
        sorted_evidences = sorted(
            qi_evidences, 
            key=lambda ev: getattr(ev, 'span_start', getattr(ev, 'start_char', 0)), 
            reverse=True
        )

        redacted_text = text
        for ev in sorted_evidences:
            start = getattr(ev, 'span_start', getattr(ev, 'start_char', 0))
            end = getattr(ev, 'span_end', getattr(ev, 'end_char', 0))
            qi_type = getattr(ev, 'qi_type', getattr(ev, 'entity_type', 'QI'))
            
            # Choose placeholder label
            placeholder = f"[{qi_type.upper()}]" if mask_format == "category" else "[REDACTED_QI]"
            
            # Replace character slice
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

        return redacted_text

    
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
        normalized_value: Optional[str],
        granularity: str,
        detector: str,
        confidence: float,
        is_unseen: bool = False,
        span_start: Optional[int] = None,
        span_end: Optional[int] = None,
    ) -> None:
        # 1. Instantiate QIEvidence using only supported __init__ fields
        qi = QIEvidence(
            qi_type=qi_type,
            normalized_value=normalized_value,
            granularity=granularity,
            detector=detector,
            detector_version=self.VERSION,
            extraction_confidence=confidence,
            is_unseen_value=is_unseen,
        )
        
        # 2. Attach span positions dynamically
        if span_start is not None:
            qi.span_start = span_start
        if span_end is not None:
            qi.span_end = span_end

        evidence.append(qi)
    
    def detect(self, text: str) -> list[QIEvidence]:
        """Detect quasi-identifiers in text with proper normalization."""
        evidence = []
        text_lower = text.lower()
        
        # === 1. AGE DETECTION (unchanged) ===
        for pattern, granularity in self.AGE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                age_str = match.group(1)
                try:
                    age = int(age_str)
                    if 0 <= age <= 120:
                        bucket_start = (age // 5) * 5
                        bucket_end = bucket_start + 4
                        self._append_qi(
                            evidence,
                            qi_type="age_5yr",
                            normalized_value=f"{bucket_start}-{bucket_end}",
                            granularity="5yr_bucket",
                            detector="pattern",
                            confidence=0.9,
                            span_start=match.start(),
                            span_end=match.end(),
                        )
                except ValueError:
                    pass
        
        # === 2. SEX DETECTION ===
        for pattern, granularity in self.SEX_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                raw_value = match.group(1).lower().strip()
                normalized = self.SEX_NORMALIZATION.get(raw_value)
                if normalized:
                    self._append_qi(
                        evidence,
                        qi_type="sex",
                        normalized_value=normalized,
                        granularity="binary",
                        detector="pattern",
                        confidence=0.85,
                        span_start=match.start(),
                        span_end=match.end(),   
                    )
        
        # === 3. MARITAL STATUS DETECTION ===
        for pattern, granularity in self.MARITAL_STATUS_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                if granularity == 'married_indicator':
                    # "my spouse/wife/husband" indicates married
                    self._append_qi(
                        evidence,
                        qi_type="marital_status",
                        normalized_value="married",
                        granularity="marital_status",
                        detector="pattern",
                        confidence=0.80,
                        span_start=match.start(),
                        span_end=match.end(),
                    )
                else:
                    raw_value = match.group(1).lower().strip()
                    normalized = self.MARITAL_STATUS_NORMALIZATION.get(raw_value)
                    if normalized:
                        self._append_qi(
                            evidence,
                            qi_type="marital_status",
                            normalized_value=normalized,
                            granularity="marital_status",
                            detector="pattern",
                            confidence=0.82,
                            span_start=match.start(),
                            span_end=match.end(),
                        )
        
        # === 4. CITIZENSHIP DETECTION ===
        for pattern, normalized_value in self.CITIZENSHIP_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="citizenship",
                    normalized_value=normalized_value,
                    granularity="citizenship_status",
                    detector="pattern",
                    confidence=0.85,
                    span_start=match.start(),
                    span_end=match.end(),
                )
        
        # === 5. RACE/ETHNICITY DETECTION ===
        for match in self.RACE_ETHNICITY_PATTERN.finditer(text_lower):
            raw_value = match.group(1).lower().strip()
            normalized = self.RACE_ETHNICITY_NORMALIZATION.get(raw_value)
            is_unseen = normalized is None
            if is_unseen:
                # Keep raw value but mark as unseen
                normalized = raw_value.replace(' ', '_')
            self._append_qi(
                evidence,
                qi_type="race_ethnicity",
                normalized_value=normalized,
                granularity="census_race_ethnicity",
                detector="pattern",
                confidence=0.78,
                span_start=match.start(),
                span_end=match.end(),
                is_unseen=is_unseen,
            )
        
        # === 6. STATE DETECTION ===
        # 6a. Full state names (O(n) where n = number of states, but dict lookup is O(1))
        for state_name, state_abbr in self.STATE_NAME_TO_ABBR.items():
            if state_name in text_lower:
                start_pos = text_lower.find(state_name)
                end_pos = start_pos + len(state_name)
                self._append_qi(
                    evidence,
                    qi_type="state",
                    normalized_value=state_abbr,
                    granularity="state",
                    detector="gazetteer",
                    confidence=0.88,
                    span_start=start_pos,
                    span_end=end_pos,
                )
        
        # 6b. Two-letter state codes in context
        for match in self.STATE_CODE_PATTERN.finditer(text):
            code = match.group(1).upper()
            if code in self.VALID_STATE_CODES:
                self._append_qi(
                    evidence,
                    qi_type="state",
                    normalized_value=code,
                    granularity="state",
                    detector="pattern",
                    confidence=0.75,
                    span_start=match.start(),
                    span_end=match.end(),
                )
        
        # 6c. spaCy NER for locations
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "GPE":
                        state_name = ent.text.strip().lower()
                        if state_name in self.STATE_NAME_TO_ABBR:
                            self._append_qi(
                                evidence,
                                qi_type="state",
                                normalized_value=self.STATE_NAME_TO_ABBR[state_name],
                                granularity="state",
                                detector="spacy_ner",
                                confidence=0.7,
                                span_start=ent.start_char,
                                span_end=ent.end_char,
                            )
            except Exception:
                pass
        
        # === 7. OCCUPATION DETECTION ===
        text_lower = text.lower()
        occupation_detected = False
        
        # Try specific keywords first
        for keyword, normalized in self.OCCUPATION_KEYWORDS.items():
            if keyword in text_lower:
                start_pos = text_lower.find(keyword)
                end_pos = start_pos + len(keyword)
                self._append_qi(
                    evidence,
                    qi_type="occupation",
                    normalized_value=normalized,
                    granularity="occupation_broad",
                    detector="keyword",
                    confidence=0.80,
                    span_start=start_pos,
                    span_end=end_pos,
                )
                occupation_detected = True
                break  # Take first match

        # === 8. DATE DETECTION ===
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="date",
                    normalized_value=match.group(0),
                    granularity="full_date",
                    detector="pattern",
                    confidence=0.85,
                    span_start=match.start(),
                    span_end=match.end(),
                )

        # === 9. TIME DETECTION ===
        for pattern in self.TIME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="time",
                    normalized_value=match.group(0),
                    granularity="time_of_day",
                    detector="pattern",
                    confidence=0.80,
                    span_start=match.start(),
                    span_end=match.end(),
                )

        # === 10. DATETIME DETECTION (combined) ===
        for pattern in self.DATETIME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="datetime",
                    normalized_value=match.group(0),
                    granularity="full_datetime",
                    detector="pattern",
                    confidence=0.90,
                    span_start=match.start(),
                    span_end=match.end(),
                )
        
        # Check for rare occupations specifically
        for term in self.RARE_OCCUPATION_TERMS:
            if term in text_lower:
                self._append_qi(
                    evidence,
                    qi_type="occupation",
                    normalized_value=term,
                    granularity="specific_occupation",
                    detector="gazetteer",
                    confidence=0.85,
                    span_start=text_lower.find(term),
                    span_end=text_lower.find(term) + len(term),
                )
        
        # === 8. RARE DISEASE DETECTION ===
        for term in self.RARE_DISEASE_TERMS:
            if term in text_lower:
                start_pos = text_lower.find(term)
                end_pos = start_pos + len(term)
                self._append_qi(
                    evidence,
                    qi_type="condition",
                    normalized_value=term.replace(' ', '_').replace('-', '_'),
                    granularity="specific_disease",
                    detector="gazetteer",
                    confidence=0.85,
                    span_start=start_pos,
                    span_end=end_pos,
                )
        
        # === 9. DATE DETECTION (unchanged) ===
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="date",
                    normalized_value=match.group(0),
                    granularity="exact_date",
                    detector="pattern",
                    confidence=0.85,
                    span_start=match.start(),
                    span_end=match.end(),
                )

        # === EDUCATION DETECTION ===
        for pattern, normalized_value in self.EDUCATION_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="education",
                    normalized_value=normalized_value,
                    granularity="education_level",
                    detector="pattern",
                    confidence=0.80,
                    span_start=match.start(),
                    span_end=match.end(),
                )
                break  # One education level per text
        
        # === EMPLOYMENT STATUS DETECTION ===
        for pattern, normalized_value in self.EMPLOYMENT_STATUS_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                self._append_qi(
                    evidence,
                    qi_type="employment_status",
                    normalized_value=normalized_value,
                    granularity="employment_status",
                    detector="pattern",
                    confidence=0.82,
                    span_start=match.start(),
                    span_end=match.end(),
                )
                break  # One employment status per text
        
        # === DEDUPLICATION ===
        # Keep one QI per (type, normalized_value) to avoid double-counting
        deduplicated: dict[tuple[str, Optional[str]], QIEvidence] = {}
        for qi in evidence:
            key = (qi.qi_type, qi.normalized_value)
            existing = deduplicated.get(key)
            if existing is None:
                deduplicated[key] = qi
                continue
            # Keep higher confidence, merge detector agreement
            if qi.extraction_confidence > existing.extraction_confidence:
                existing.extraction_confidence = qi.extraction_confidence
            if qi.detector and qi.detector != existing.detector:
                if hasattr(existing, 'detector_agreement') and existing.detector_agreement is not None:
                    if qi.detector not in existing.detector_agreement:
                        existing.detector_agreement.append(qi.detector)
        
        return list(deduplicated.values())
    
    def check_assertion_status(self, text: str, qi: QIEvidence) -> QIEvidence:
        """
        Check if a QI is negated, hypothetical, or about someone other than patient.
        """
        # Simple negation check
        negation_patterns = [
            r'\b(?:no|not|never|without|denies|denied|negative)\b',
        ]
        
        # Check context around the QI value
        if qi.normalized_value:
            for pattern in negation_patterns:
                # Look for negation within 50 chars before the value
                value_pos = text.lower().find(qi.normalized_value.lower().replace('_', ' '))
                if value_pos > 0:
                    context = text[max(0, value_pos - 50):value_pos].lower()
                    if re.search(pattern, context):
                        qi.assertion_status = "negated"
                        break
        
        return qi