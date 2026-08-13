"""
detectors.py - Fast deterministic PII and QI detection
"""

import re
from typing import Optional

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
except Exception:  # pragma: no cover
    AnalyzerEngine = None  # type: ignore[assignment]
    NlpEngineProvider = None  # type: ignore[assignment]

from policy import (
    PIIEvidence, QIEvidence, EntityCategory, PolicyProfile
)


class PIIDetector:
    """
    Fast PII detection using Presidio + custom patterns + spaCy fallback.
    """
    
    VERSION = "1.0.0"
    
    CUSTOM_PATTERNS = {
        "US_SSN": [r'\b\d{3}-\d{2}-\d{4}\b'],
        "EMAIL_ADDRESS": [r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'],
        "PHONE_NUMBER": [r'\b(?:\+?1[-.\s]*)?(?:\(?\d{3}\)?[-.\s]*)\d{3}[-.\s]*\d{4}\b'],
        "MEDICAL_RECORD_NUMBER": [
            r'\bMRN[:\s#]*\d{6,12}\b',
            r'\bMR[:\s#]*\d{6,12}\b',
            r'\bPatient\s*ID[:\s#]*\d{6,12}\b',
        ],
        "HEALTH_PLAN_ID": [r'\b[A-Z]{3}\d{9,12}\b'],
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
        self._spacy_nlp = None

    def _create_analyzer(self) -> Optional[AnalyzerEngine]:
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

    def _detect_with_spacy(self, text: str) -> list[dict]:
        results = []
        if self._spacy_nlp is None:
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
        results = []
        for entity_type, patterns in self.CUSTOM_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    results.append({
                        'entity_type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'score': 0.9,
                        'detector': 'custom_regex'
                    })
        return results

    def detect(self, text: str, policy: PolicyProfile) -> list[PIIEvidence]:
        evidence = []
        
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

        for r in self._detect_custom_patterns(text):
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

        return self._merge_overlapping(evidence)

    def _merge_overlapping(self, evidence: list[PIIEvidence]) -> list[PIIEvidence]:
        if not evidence:
            return []
        
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
            if current.span_start < last.span_end:
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
                        confirmed_by=[last.detector] if hasattr(last, 'detector') else [],
                        merged_from_overlapping=True,
                    )
                else:
                    if hasattr(merged[-1], 'confirmed_by') and isinstance(merged[-1].confirmed_by, list):
                        merged[-1].confirmed_by.append(current.detector)
                    merged[-1].span_end = max(last.span_end, current.span_end)
                    merged[-1].merged_from_overlapping = True
            else:
                merged.append(current)
        
        return merged


class QIDetector:
    """
    Detects quasi-identifiers in text with normalization and accurate span offsets.
    """
    
    VERSION = "1.1.0"
    
    SEX_PATTERNS = [
        (r'\b(?:sex|gender)\s*[:=]?\s*(male|female|man|woman|boy|girl)\b', 'contextual'),
        (r'\b(?:i\s+am\s+)?(male|female|a\s+man|a\s+woman)\b', 'statement'),
        (r'\b(male|female)\b', 'standalone'),
    ]
    
    SEX_NORMALIZATION = {
        'male': 'male', 'man': 'male', 'boy': 'male', 'a man': 'male',
        'female': 'female', 'woman': 'female', 'girl': 'female', 'a woman': 'female',
    }
    
    MARITAL_STATUS_PATTERNS = [
        (r"\b(?:marital\s+status\s*[:=]?\s*)?(married|single|divorced|widowed|separated|never\s+married|never\s+been\s+married|unmarried|i'?m\s+married|i'?m\s+single|i'?m\s+divorced|i'?m\s+widowed|i'?m\s+separated)\b", 'contextual'),
        (r"\bmy\s+(?:spouse|wife|husband)\b", 'married_indicator'),
    ]
    
    MARITAL_STATUS_NORMALIZATION = {
        'married': 'married', "i'm married": 'married', 'im married': 'married',
        'single': 'single', "i'm single": 'single', 'im single': 'single',
        'never married': 'single', 'never been married': 'single', 'unmarried': 'single',
        'divorced': 'divorced', "i'm divorced": 'divorced', 'im divorced': 'divorced',
        'widowed': 'widowed', "i'm widowed": 'widowed', 'im widowed': 'widowed',
        'separated': 'separated', "i'm separated": 'separated', 'im separated': 'separated',
    }
    
    CITIZENSHIP_PATTERNS = [
        (r'\b(?:u\.?s\.?\s+citizen\s+by\s+naturalization|citizen\s+by\s+naturalization|naturalized\s+citizen|naturalized\s+u\.?s\.?\s+citizen|became\s+a\s+citizen|citizenship\s+through\s+naturalization)\b', 'naturalized'),
        (r'\b(?:born\s+in\s+(?:the\s+)?(?:united\s+states|u\.?s\.?|usa|puerto\s+rico|guam|u\.?s\.?\s+virgin\s+islands|american\s+samoa)|us[-\s]?born|native[-\s]?born\s+citizen)\b', 'us_born'),
        (r'\b(?:u\.?s\.?\s+citizen|american\s+citizen|citizen\s+of\s+(?:the\s+)?united\s+states|i\s+am\s+a\s+citizen)\b', 'citizen'),
        (r'\b(?:not\s+a\s+citizen|not\s+a\s+u\.?s\.?\s+citizen|i\s+am\s+not\s+a\s+citizen|noncitizen|non[-\s]?citizen|foreign[-\s]?national|foreign[-\s]?born(?!\s+citizen))\b', 'noncitizen'),
        (r'\bcitizenship\s+status\s+is\s+not\s+a\s+citizen\b', 'noncitizen'),
    ]
    
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
    
    RACE_ETHNICITY_NORMALIZATION = {
        'white': 'white', 'caucasian': 'white',
        'black': 'black', 'african american': 'black', 'african-american': 'black',
        'asian': 'asian', 'asian indian': 'asian_indian', 'chinese': 'chinese',
        'korean': 'korean', 'japanese': 'japanese', 'vietnamese': 'vietnamese',
        'filipino': 'filipino', 'filipina': 'filipino', 'cambodian': 'cambodian',
        'laotian': 'laotian', 'hmong': 'hmong',
        'pacific islander': 'pacific_islander', 'native hawaiian': 'native_hawaiian',
        'samoan': 'samoan', 'tongan': 'tongan',
        'hispanic': 'hispanic', 'latino': 'hispanic', 'latina': 'hispanic',
        'latinx': 'hispanic', 'mexican': 'mexican', 'puerto rican': 'puerto_rican', 'cuban': 'cuban',
        'native american': 'native_american', 'american indian': 'native_american',
        'alaska native': 'alaska_native', 'indigenous': 'native_american',
        'two or more races': 'two_or_more_races', 'multiracial': 'two_or_more_races', 'mixed race': 'two_or_more_races',
    }
    
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
    
    VALID_STATE_CODES = set(STATE_NAME_TO_ABBR.values())
    STATE_CODE_PATTERN = re.compile(r'\b(?:in|from|living\s+in|reside\s+in|state\s+of|,)\s*([A-Z]{2})\b')
    
    OCCUPATION_KEYWORDS = {
        'doctor': 'healthcare_practitioners', 'physician': 'healthcare_practitioners',
        'nurse': 'healthcare_practitioners', 'registered nurse': 'registered_nurse',
        'medical': 'healthcare_practitioners', 'hospital': 'healthcare_practitioners',
        'healthcare': 'healthcare_practitioners', 'pharmacist': 'healthcare_practitioners',
        'therapist': 'healthcare_practitioners', 'surgeon': 'healthcare_practitioners',
        'dentist': 'healthcare_practitioners',
        'teacher': 'education_training_library', 'professor': 'postsecondary_teacher',
        'educator': 'education_training_library', 'instructor': 'education_training_library',
        'school': 'education_training_library',
        'zoologist': 'zoologist', 'astronomer': 'astronomer', 'epidemiologist': 'epidemiologist',
    }
    
    RARE_OCCUPATION_TERMS = {'zoologist', 'astronomer', 'epidemiologist'}
    RARE_DISEASE_TERMS = {'ehlers-danlos', 'ehlers danlos', 'huntington', 'huntingtons', 'marfan', 'cystic fibrosis'}
    
    AGE_PATTERNS = [
        (r'\b(?:age|aged)\s*[:=]?\s*(\d{1,3})\b', 'contextual'),
        (r'\b(\d{1,3})\s*(?:years?\s+old|y/?o|yo)\b', 'contextual'),
        (r"\bI(?:'m|am)\s+(\d{1,3})\b", 'statement'),
    ]
    
    DATE_PATTERNS = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
    ]

    TIME_PATTERNS = [
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?\b',
        r'\b\d{1,2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b',
        r'\b(?:at\s+)?\d{1,2}:\d{2}(?::\d{2})?\b',
        r'\b(?:noon|midnight)\b',
        r'\b\d{1,2}\s*(?:o\'?clock)\b',
    ]

    DATETIME_PATTERNS = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b',
    ]

    EDUCATION_PATTERNS = [
        (r'\b(?:completed?|have|earned?|holds?|with)\s+(?:a\s+)?(high school|hs)\s*(?:diploma|degree)?\b', 'high_school_diploma'),
        (r'\b(?:ged|g\.e\.d\.)\b', 'ged'),
        (r'\bsome\s+college\b', 'some_college_1yr_plus'),
        (r'\b(?:associate[\'s]?s?|aa|as)\s*(?:degree)?\b', 'associates_degree'),
        (r'\b(?:bachelor[\'s]?s?|ba|bs|b\.a\.|b\.s\.)\s*(?:degree)?\b', 'bachelors_degree'),
        (r'\b(?:master[\'s]?s?|ma|ms|mba|m\.a\.|m\.s\.)\s*(?:degree)?\b', 'masters_degree'),
        (r'\b(?:doctorate|doctoral|ph\.?d\.?|phd)\b', 'doctorate_degree'),
    ]
    
    EMPLOYMENT_STATUS_PATTERNS = [
        (r'\b(?:currently\s+)?(?:employed|working|at\s+work)\b', 'employed_at_work'),
        (r'\b(?:unemployed|out\s+of\s+work|jobless|looking\s+for\s+(?:a\s+)?(?:job|work))\b', 'unemployed'),
        (r'\b(?:retired|not\s+working)\b', 'not_in_labor_force'),
        (r'\b(?:in\s+the\s+)?(?:military|armed\s+forces|army|navy|air\s+force|marines?|coast\s+guard)\b', 'armed_forces_at_work'),
    ]

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
        evidence.append(QIEvidence(
            qi_type=qi_type,
            normalized_value=normalized_value,
            granularity=granularity,
            detector=detector,
            detector_version=self.VERSION,
            extraction_confidence=confidence,
            is_unseen_value=is_unseen,
            span_start=span_start,
            span_end=span_end,
        ))

    def redact_qi_text(self, text: str, qi_evidences: list[QIEvidence], mask_format="category") -> str:
        """
        Redacts detected quasi-identifiers from the text using detected QIEvidence offsets.
        """
        valid_evidences = [
            ev for ev in qi_evidences 
            if getattr(ev, 'span_start', None) is not None and getattr(ev, 'span_end', None) is not None
        ]

        if not valid_evidences:
            return text

        # Sort in reverse order by start offset so text replacements don't shift offsets
        sorted_evidences = sorted(
            valid_evidences, 
            key=lambda ev: ev.span_start, 
            reverse=True
        )

        redacted_text = text
        for ev in sorted_evidences:
            start = ev.span_start
            end = ev.span_end
            placeholder = f"[{ev.qi_type.upper()}]" if mask_format == "category" else "[REDACTED_QI]"
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

        return redacted_text

    def detect(self, text: str) -> list[QIEvidence]:
        evidence = []
        text_lower = text.lower()
        
        # 1. AGE
        for pattern, granularity in self.AGE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    age = int(match.group(1))
                    if 0 <= age <= 120:
                        bucket_start = (age // 5) * 5
                        self._append_qi(
                            evidence, qi_type="age_5yr",
                            normalized_value=f"{bucket_start}-{bucket_start+4}",
                            granularity="5yr_bucket", detector="pattern",
                            confidence=0.9, span_start=match.start(), span_end=match.end()
                        )
                except ValueError:
                    pass
        
        # 2. SEX
        for pattern, granularity in self.SEX_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                raw_val = match.group(1).lower().strip()
                normalized = self.SEX_NORMALIZATION.get(raw_val)
                if normalized:
                    self._append_qi(
                        evidence, qi_type="sex", normalized_value=normalized,
                        granularity="binary", detector="pattern", confidence=0.85,
                        span_start=match.start(), span_end=match.end()
                    )
        
        # 3. MARITAL STATUS
        for pattern, granularity in self.MARITAL_STATUS_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                if granularity == 'married_indicator':
                    self._append_qi(
                        evidence, qi_type="marital_status", normalized_value="married",
                        granularity="marital_status", detector="pattern", confidence=0.80,
                        span_start=match.start(), span_end=match.end()
                    )
                else:
                    raw_val = match.group(1).lower().strip()
                    normalized = self.MARITAL_STATUS_NORMALIZATION.get(raw_val)
                    if normalized:
                        self._append_qi(
                            evidence, qi_type="marital_status", normalized_value=normalized,
                            granularity="marital_status", detector="pattern", confidence=0.82,
                            span_start=match.start(), span_end=match.end()
                        )
        
        # 4. CITIZENSHIP
        for pattern, normalized_val in self.CITIZENSHIP_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                self._append_qi(
                    evidence, qi_type="citizenship", normalized_value=normalized_val,
                    granularity="citizenship_status", detector="pattern", confidence=0.85,
                    span_start=match.start(), span_end=match.end()
                )
        
        # 5. RACE/ETHNICITY
        for match in self.RACE_ETHNICITY_PATTERN.finditer(text_lower):
            raw_val = match.group(1).lower().strip()
            normalized = self.RACE_ETHNICITY_NORMALIZATION.get(raw_val)
            is_unseen = normalized is None
            if is_unseen:
                normalized = raw_val.replace(' ', '_')
            self._append_qi(
                evidence, qi_type="race_ethnicity", normalized_value=normalized,
                granularity="census_race_ethnicity", detector="pattern", confidence=0.78,
                is_unseen=is_unseen, span_start=match.start(), span_end=match.end()
            )
        
        # 6. STATE
        for state_name, state_abbr in self.STATE_NAME_TO_ABBR.items():
            pattern = r'\b' + re.escape(state_name) + r'\b'
            for match in re.finditer(pattern, text_lower):
                self._append_qi(
                    evidence, qi_type="state", normalized_value=state_abbr,
                    granularity="state", detector="gazetteer", confidence=0.88,
                    span_start=match.start(), span_end=match.end()
                )
        
        for match in self.STATE_CODE_PATTERN.finditer(text):
            code = match.group(1).upper()
            if code in self.VALID_STATE_CODES:
                self._append_qi(
                    evidence, qi_type="state", normalized_value=code,
                    granularity="state", detector="pattern", confidence=0.75,
                    span_start=match.start(1), span_end=match.end(1)
                )

        # 7. OCCUPATION
        for keyword, normalized in self.OCCUPATION_KEYWORDS.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, text_lower):
                self._append_qi(
                    evidence, qi_type="occupation", normalized_value=normalized,
                    granularity="occupation_broad", detector="keyword", confidence=0.80,
                    span_start=match.start(), span_end=match.end()
                )

        # 8. DATE, TIME, DATETIME
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self._append_qi(
                    evidence, qi_type="date", normalized_value=match.group(0),
                    granularity="full_date", detector="pattern", confidence=0.85,
                    span_start=match.start(), span_end=match.end()
                )

        for pattern in self.TIME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self._append_qi(
                    evidence, qi_type="time", normalized_value=match.group(0),
                    granularity="time_of_day", detector="pattern", confidence=0.80,
                    span_start=match.start(), span_end=match.end()
                )

        for pattern in self.DATETIME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self._append_qi(
                    evidence, qi_type="datetime", normalized_value=match.group(0),
                    granularity="full_datetime", detector="pattern", confidence=0.90,
                    span_start=match.start(), span_end=match.end()
                )

        # 9. RARE DISEASES
        for term in self.RARE_DISEASE_TERMS:
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text_lower):
                self._append_qi(
                    evidence, qi_type="condition",
                    normalized_value=term.replace(' ', '_').replace('-', '_'),
                    granularity="specific_disease", detector="gazetteer", confidence=0.85,
                    span_start=match.start(), span_end=match.end()
                )

        # 10. EDUCATION & EMPLOYMENT
        for pattern, normalized_val in self.EDUCATION_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                self._append_qi(
                    evidence, qi_type="education", normalized_value=normalized_val,
                    granularity="education_level", detector="pattern", confidence=0.80,
                    span_start=match.start(), span_end=match.end()
                )

        for pattern, normalized_val in self.EMPLOYMENT_STATUS_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                self._append_qi(
                    evidence, qi_type="employment_status", normalized_value=normalized_val,
                    granularity="employment_status", detector="pattern", confidence=0.82,
                    span_start=match.start(), span_end=match.end()
                )

        # Deduplicate strictly by position span to preserve distinct redaction targets
        deduplicated: dict[tuple[int, int, str], QIEvidence] = {}
        for qi in evidence:
            key = (qi.span_start, qi.span_end, qi.qi_type)
            if key not in deduplicated or qi.extraction_confidence > deduplicated[key].extraction_confidence:
                deduplicated[key] = qi

        return list(deduplicated.values())

    def check_assertion_status(self, text: str, qi: QIEvidence) -> QIEvidence:
        """
        Check if a QI is negated in context preceding its specific span.
        """
        if qi.span_start is None:
            return qi

        negation_patterns = [r'\b(?:no|not|never|without|denies|denied|negative)\b']
        context_start = max(0, qi.span_start - 50)
        context = text[context_start:qi.span_start].lower()

        for pattern in negation_patterns:
            if re.search(pattern, context):
                qi.assertion_status = "negated"
                break

        return qi