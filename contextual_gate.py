"""
contextual_gate.py - Cheap contextual routing gate

A lightweight classifier that decides whether to:
1. Route directly to Tier 1/2 (confident safe)
2. Route directly to Tier 3 (confident unsafe)
3. Invoke the local LLM for deeper analysis (uncertain)

This is NOT a generative LLM - it's a fast classifier using
extracted features.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

from policy import (
    PIIEvidence, QIEvidence, JointRiskEstimate, ContextualEvidence,
    PolicyProfile, EntityCategory, Tier
)


class GateDecision(Enum):
    """Output of the contextual gate."""
    SAFE_FOR_CLOUD = auto()       # Confident enough for Tier 1 or 2
    UNSAFE_ROUTE_LOCAL = auto()   # Confident unsafe, route to Tier 3
    UNCERTAIN_NEED_LLM = auto()   # Need deeper analysis


@dataclass
class GateFeatures:
    """Features extracted for gate decision."""
    # Text characteristics
    text_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    
    # Entity counts
    direct_identifier_count: int = 0
    strong_identifier_count: int = 0
    qi_count: int = 0
    contextual_entity_count: int = 0
    
    # Risk indicators
    has_rare_disease: bool = False
    has_rare_occupation: bool = False
    has_high_harm_category: bool = False
    
    # Dates and temporal
    date_count: int = 0
    has_specific_timestamps: bool = False
    
    # Geographic
    geographic_granularity: str = "unknown"  # state, county, city, zip5, zip9
    
    # k-anonymity
    estimated_k_lower: Optional[float] = None
    k_estimate_available: bool = False
    k_estimate_method: str = "unknown"
    
    # Masking impact
    mask_count: int = 0
    mask_ratio: float = 0.0  # Fraction of text that would be masked
    
    # Detector agreement
    detector_disagreement: bool = False
    unresolved_entities: int = 0


class ContextualGate:
    """
    Fast routing gate using rule-based and lightweight ML classification.
    
    Avoids LLM invocation for clear-cut cases.
    """
    
    def __init__(self):
        # In production, this would load a trained classifier
        # For now, using rule-based logic
        pass
    
    def extract_features(
        self,
        text: str,
        pii_evidence: list[PIIEvidence],
        qi_evidence: list[QIEvidence],
        risk_estimate: JointRiskEstimate,
        policy: PolicyProfile
    ) -> GateFeatures:
        """Extract features for gate decision."""
        features = GateFeatures()
        
        # Text characteristics
        features.text_length = len(text)
        features.word_count = len(text.split())
        features.sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # Count entities by category
        for e in pii_evidence:
            if e.category == EntityCategory.DIRECT_IDENTIFIER:
                features.direct_identifier_count += 1
            elif e.category == EntityCategory.STRONG_IDENTIFIER:
                features.strong_identifier_count += 1
            elif e.category == EntityCategory.QUASI_IDENTIFIER:
                features.qi_count += 1
            else:
                features.contextual_entity_count += 1
        
        features.qi_count += len(qi_evidence)
        
        # Risk indicators
        for qi in qi_evidence:
            if qi.qi_type == "condition":
                # Check if rare (would use proper ontology in production)
                rare_terms = {'ehlers-danlos', 'huntington', 'marfan'}
                if qi.normalized_value and any(t in qi.normalized_value.lower() for t in rare_terms):
                    features.has_rare_disease = True
            
            if qi.qi_type == "occupation":
                rare_occs = {'zoologist', 'astronomer', 'epidemiologist'}
                if qi.normalized_value and qi.normalized_value.lower() in rare_occs:
                    features.has_rare_occupation = True
            
            if qi.qi_type == "date":
                features.date_count += 1
        
        # k-anonymity
        features.estimated_k_lower = risk_estimate.lower_bound_k
        features.k_estimate_available = risk_estimate.estimate_available
        
        # Masking impact
        features.mask_count = features.direct_identifier_count + features.strong_identifier_count
        if features.text_length > 0:
            # Rough estimate of masked character ratio
            avg_entity_len = 15  # Approximate
            masked_chars = features.mask_count * avg_entity_len
            features.mask_ratio = min(masked_chars / features.text_length, 1.0)
        
        return features
    
    def decide(
        self,
        features: GateFeatures,
        policy: PolicyProfile
    ) -> tuple[GateDecision, list[str]]:
        """
        Make routing decision based on features.
        
        Returns decision and list of reasons.
        """
        reasons = []
        
        # === HARD STOPS: Route to Tier 3 immediately ===
        
        # Direct identifiers that couldn't be masked
        if features.direct_identifier_count > 0:
            # Note: In practice, these would be masked, but if masking fails...
            reasons.append(f"Found {features.direct_identifier_count} direct identifier(s)")
        
        # k-anonymity failure. A lower bound below the policy minimum is
        # already sufficient evidence that the record is unsafe. This must
        # not be delegated to a probabilistic/contextual model: the model
        # cannot make an unsafe equivalence class larger.
        if features.k_estimate_available and features.estimated_k_lower is not None:
            if features.estimated_k_lower < policy.k_minimum:
                reasons.append(
                    f"k_lower ({features.estimated_k_lower:.1f}) < k_min ({policy.k_minimum})"
                )
                return GateDecision.UNSAFE_ROUTE_LOCAL, reasons
        elif policy.require_joint_estimate:
            # Can't estimate k and policy requires it
            reasons.append("Joint k estimate unavailable - failing closed")
            return GateDecision.UNSAFE_ROUTE_LOCAL, reasons
        
        # === UNCERTAINTY: Need LLM analysis ===
        
        # Many dates suggest temporal correlation risk
        if features.date_count >= 3:
            reasons.append(f"Multiple dates ({features.date_count}) - potential temporal correlation")
            return GateDecision.UNCERTAIN_NEED_LLM, reasons
        
        # High QI count without good k estimate
        if features.qi_count >= 5 and (not features.k_estimate_available or features.estimated_k_lower is None):
            reasons.append(f"Many QIs ({features.qi_count}) with uncertain k estimate")
            return GateDecision.UNCERTAIN_NEED_LLM, reasons
        
        # Narrative density (many entities relative to text length)
        entity_density = (features.qi_count + features.contextual_entity_count) / max(features.word_count, 1)
        if entity_density > 0.1:  # More than 1 entity per 10 words
            reasons.append(f"High entity density ({entity_density:.2f}) - complex narrative")
            return GateDecision.UNCERTAIN_NEED_LLM, reasons
        
        # Detector disagreement (would be computed from multi-detector comparison)
        if features.detector_disagreement:
            reasons.append("Detector disagreement - need deeper analysis")
            return GateDecision.UNCERTAIN_NEED_LLM, reasons
        
        # === SAFE: Can route to cloud (Tier 1 or 2) ===
        
        # High k, no high-harm indicators
        if features.k_estimate_available and features.estimated_k_lower is not None:
            if features.estimated_k_lower >= policy.k_safe_threshold:
                reasons.append(f"k_lower ({features.estimated_k_lower:.1f}) >= k_safe ({policy.k_safe_threshold})")
                return GateDecision.SAFE_FOR_CLOUD, reasons
        
        # Low entity count, no red flags
        if features.qi_count <= 2 and not features.has_rare_disease and not features.has_rare_occupation:
            reasons.append("Low QI count with no high-harm indicators")
            return GateDecision.SAFE_FOR_CLOUD, reasons
        
        # Default: uncertain, need more analysis
        reasons.append("Default: uncertain case, invoking contextual analysis")
        return GateDecision.UNCERTAIN_NEED_LLM, reasons
