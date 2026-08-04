"""
policy.py - Policy profiles and core data structures

A policy profile explicitly defines:
- What population is being protected
- What auxiliary data an attacker might have
- Minimum acceptable k-anonymity threshold
- High-harm categories requiring automatic local processing
- Acceptable false-cloud-release rate
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, FrozenSet
from datetime import datetime


class Tier(Enum):
    """Routing destination for text."""
    TIER_1_CLOUD_ORIGINAL = auto()  # No sensitive content detected
    TIER_2_CLOUD_MASKED = auto()    # Identifiers masked, residual risk bounded
    TIER_3_LOCAL = auto()           # Must process locally


class HarmCategory(Enum):
    """Categories of sensitive information requiring special handling."""
    PSYCHIATRIC = "psychiatric"
    SUBSTANCE_USE = "substance_use"
    REPRODUCTIVE = "reproductive"
    GENETIC = "genetic"
    HIV_STI = "hiv_sti"
    ABUSE_VIOLENCE = "abuse_violence"
    RARE_DISEASE = "rare_disease"
    EMPLOYMENT_SENSITIVE = "employment_sensitive"
    LEGAL_IMMIGRATION = "legal_immigration"


class EntityCategory(Enum):
    """Broad categories for detected entities."""
    DIRECT_IDENTIFIER = "direct_identifier"  # SSN, MRN, etc - always mask
    STRONG_IDENTIFIER = "strong_identifier"  # Name, email, phone
    QUASI_IDENTIFIER = "quasi_identifier"    # Age, location, condition
    CONTEXTUAL = "contextual"                # Dates, organizations


@dataclass(frozen=True)
class PolicyProfile:
    """
    Defines the privacy policy for routing decisions.
    
    This should be configured per deployment, not hardcoded.
    """
    name: str
    version: str
    
    # Population and attacker model
    reference_population: str  # e.g., "hospital_2024", "state_medicaid"
    attacker_auxiliary_data: FrozenSet[str] = frozenset()  # e.g., {"voter_rolls", "linkedin"}
    
    # k-anonymity requirements
    k_minimum: int = 5  # Minimum acceptable equivalence class size
    k_safe_threshold: int = 20  # Above this, don't invoke contextual analysis
    
    # High-harm categories - always route to Tier 3
    high_harm_categories: FrozenSet[HarmCategory] = frozenset({
        HarmCategory.PSYCHIATRIC,
        HarmCategory.SUBSTANCE_USE,
        HarmCategory.REPRODUCTIVE,
        HarmCategory.GENETIC,
        HarmCategory.HIV_STI,
    })
    
    # Direct identifiers - always mask, regardless of other factors
    direct_identifier_types: FrozenSet[str] = frozenset({
        "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "MEDICAL_RECORD_NUMBER",
        "US_ITIN", "CREDIT_CARD", "US_BANK_NUMBER", "HEALTH_PLAN_ID",
    })
    
    # Cloud processing permissions
    cloud_allows_phi: bool = False
    cloud_allows_masked_phi: bool = True
    
    # Acceptable error rates
    max_false_cloud_rate: float = 0.01  # 1% false release tolerance
    
    # Operational
    require_joint_estimate: bool = True  # Fail if joint k cannot be estimated
    contextual_review_threshold: float = 0.3  # Gate threshold for LLM review


@dataclass
class PIIEvidence:
    """Evidence for a single PII/PHI detection."""
    entity_type: str
    category: EntityCategory
    span_start: int
    span_end: int
    detector: str
    detector_version: str
    raw_detector_score: float
    calibrated_probability: Optional[float]  # Actual detection probability, if calibrated
    confirmed_by: list[str] = field(default_factory=list)  # Other detectors agreeing
    harm_categories: list[HarmCategory] = field(default_factory=list)
    masking_action: str = "none"  # "masked", "generalized", "suppressed", "none"
    preserves_utility: Optional[bool] = None
    merged_from_overlapping: bool = False


@dataclass 
class QIEvidence:
    """Evidence for a single quasi-identifier."""
    qi_type: str  # age, sex, location, occupation, condition, etc.
    normalized_value: Optional[str]  # Ontology code or standardized value
    granularity: str  # e.g., "exact_age", "5yr_bucket", "state", "zip3"
    
    # Clinical context (important for medical QIs)
    assertion_status: str = "present"  # present, negated, hypothetical, conditional
    experiencer: str = "patient"  # patient, family, clinician, other
    temporal_status: str = "current"  # current, historical, future
    
    # Detection metadata
    detector: str = ""
    detector_version: str = ""
    extraction_confidence: float = 0.0
    calibrated_extraction_prob: Optional[float] = None
    detector_agreement: list[str] = field(default_factory=list)
    
    # Population frequency data
    frequency_source: Optional[str] = None
    frequency_vintage: Optional[str] = None  # e.g., "ACS_2022"
    marginal_frequency: Optional[float] = None
    frequency_lower_bound: Optional[float] = None
    frequency_upper_bound: Optional[float] = None
    is_unseen_value: bool = False  # Value not in reference data


@dataclass
class JointRiskEstimate:
    """Estimated joint re-identification risk for a QI combination."""
    qi_combination: tuple[str, ...]  # Which QIs were combined
    method: str  # "empirical_joint", "model_based", "conservative_bound", "unavailable"
    
    # Estimates
    point_estimate_k: Optional[float] = None
    lower_bound_k: Optional[float] = None
    upper_bound_k: Optional[float] = None
    
    # Metadata
    reference_population: str = ""
    population_size: Optional[int] = None
    estimation_confidence: float = 0.0
    
    # Flags
    used_independence_assumption: bool = False
    has_unseen_values: bool = False
    estimate_available: bool = True


@dataclass
class ContextualEvidence:
    """Structured flags from contextual analysis (not a single score)."""
    # Event indicators
    unusual_event: bool = False
    unusual_event_confidence: float = 0.0
    public_searchable_event: bool = False
    public_event_confidence: float = 0.0
    
    # Community/population indicators
    small_community: bool = False
    small_community_confidence: float = 0.0
    
    # Correlation risks
    temporal_correlation_risk: bool = False
    temporal_confidence: float = 0.0
    relationship_network_risk: bool = False
    relationship_confidence: float = 0.0
    
    # Inference risks
    inferential_medical_disclosure: bool = False
    inferential_confidence: float = 0.0
    rare_combination_indicator: bool = False
    
    # Analysis metadata
    analysis_performed: bool = False
    model_abstained: bool = False
    parsing_error: bool = False
    overall_confidence: float = 0.0
    evidence_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Complete record of a routing decision."""
    # Decision
    tier: Tier
    decision_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Policy context
    policy_name: str = ""
    policy_version: str = ""
    reference_population_version: str = ""
    detector_versions: dict = field(default_factory=dict)
    
    # Reasoning
    hard_stop_reasons: list[str] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    contextual_review_invoked: bool = False
    
    # Risk summary (for logging, not for decision-making)
    estimated_k_lower: Optional[float] = None
    high_harm_detected: bool = False
    direct_identifiers_found: int = 0
    qis_found: int = 0
    
    # Input tracking (no raw text!)
    input_hash: str = ""  # SHA256 for deduplication
    input_length: int = 0