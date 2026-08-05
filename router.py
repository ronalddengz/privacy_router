"""
router.py - Main privacy-preserving text router

Implements the pipeline:
1. Fast PII/QI detection
2. Local joint risk lookup
3. Deterministic routing for clear cases
4. Contextual gate for uncertain cases
5. Local LLM only when gate is uncertain
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from pathlib import Path

from policy import (
    PolicyProfile, Tier, RoutingDecision, PIIEvidence, QIEvidence,
    JointRiskEstimate, ContextualEvidence, EntityCategory
)
from detectors import PIIDetector, QIDetector
from risk_estimator import JointRiskEstimator
from contextual_gate import ContextualGate, GateDecision, GateFeatures
from contextual_llm import ContextualLLMAnalyzer
from frequency_tables import LocalFrequencyTable


@dataclass
class RoutingResult:
    """Complete result of routing analysis."""
    decision: RoutingDecision
    masked_text: str
    
    # Evidence (kept locally, not sent to cloud)
    pii_evidence: list[PIIEvidence] = field(default_factory=list)
    qi_evidence: list[QIEvidence] = field(default_factory=list)
    risk_estimate: Optional[JointRiskEstimate] = None
    contextual_evidence: Optional[ContextualEvidence] = None
    gate_features: Optional[GateFeatures] = None
    llm_time: float = 0.0


class PrivacyRouter:
    """
    Main router implementing the privacy-preserving pipeline.
    
    Design principles:
    - Fast path first (no LLM for clear cases)
    - Evidence over scores
    - Fail closed on uncertainty
    - Local LLM is exception handler, not default
    """
    
    def __init__(
        self,
        policy: PolicyProfile,
        frequency_table: LocalFrequencyTable,
        enable_contextual_llm: bool = True
    ):
        self.policy = policy
        self.freq_table = frequency_table
        self.enable_llm = enable_contextual_llm
        
        # Initialize components
        self.pii_detector = PIIDetector()
        self.qi_detector = QIDetector()
        self.risk_estimator = JointRiskEstimator(frequency_table)
        self.gate = ContextualGate()
        self.llm_analyzer = ContextualLLMAnalyzer() if enable_contextual_llm else None
    
    def route(self, text: str) -> RoutingResult:
        """
        Route text to appropriate tier based on privacy analysis.
        
        Returns masked text and routing decision with full evidence.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        input_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        llm_time = 0.0
        
        # === Stage 1: Fast PII/PHI Detection ===
        pii_evidence = self.pii_detector.detect(text, self.policy)
        
        # === Stage 2: QI Detection ===
        qi_evidence = self.qi_detector.detect(text)
        
        # Check assertion status for medical QIs
        qi_evidence = [
            self.qi_detector.check_assertion_status(text, qi)
            for qi in qi_evidence
        ]
        
        # === Stage 3: Mask Direct and Strong Identifiers ===
        masked_text = self._apply_masking(text, pii_evidence)
        
        # === Stage 4: Joint Risk Estimation ===
        risk_estimate = self.risk_estimator.estimate(qi_evidence, self.policy)
        
        # === Stage 5: Check High-Harm Categories ===
        high_harm = self.risk_estimator.check_high_harm_qis(qi_evidence, self.policy)
        
        # === Stage 6: Deterministic Hard Stops ===
        hard_stops = []
        
        # Direct identifier found and couldn't mask (edge case)
        unmasked_direct = [
            e for e in pii_evidence
            if e.category == EntityCategory.DIRECT_IDENTIFIER and e.masking_action == "none"
        ]
        if unmasked_direct:
            hard_stops.append(f"Unmasked direct identifiers: {len(unmasked_direct)}")
        
        # High-harm category
        if high_harm:
            categories = [str(h[1].value) for h in high_harm]
            hard_stops.append(f"High-harm categories: {', '.join(categories)}")
        
        # k below minimum with high confidence
        if risk_estimate.estimate_available and risk_estimate.lower_bound_k is not None:
            if risk_estimate.lower_bound_k < self.policy.k_minimum:
                if risk_estimate.method == "empirical_joint":
                    hard_stops.append(
                        f"k_lower ({risk_estimate.lower_bound_k:.1f}) < k_min ({self.policy.k_minimum})"
                    )
        
        # Cannot estimate k and policy requires it
        if not risk_estimate.estimate_available and self.policy.require_joint_estimate:
            hard_stops.append("Joint k estimate unavailable")
        
        # Any hard stop is a definitive local-only decision. Previously this
        # branch required high_harm as well, allowing unmasked identifiers,
        # low-k records, and unavailable required estimates to proceed to the
        # contextual gate.
        if hard_stops:
            decision = RoutingDecision(
                tier=Tier.TIER_3_LOCAL,
                policy_name=self.policy.name,
                policy_version=self.policy.version,
                hard_stop_reasons=hard_stops,
                estimated_k_lower=risk_estimate.lower_bound_k,
                high_harm_detected=bool(high_harm),
                direct_identifiers_found=len([
                    e for e in pii_evidence 
                    if e.category == EntityCategory.DIRECT_IDENTIFIER
                ]),
                qis_found=len(qi_evidence),
                input_hash=input_hash,
                input_length=len(text),
                contextual_review_invoked=False,
            )
            return RoutingResult(
                decision=decision,
                masked_text=masked_text,
                pii_evidence=pii_evidence,
                qi_evidence=qi_evidence,
                risk_estimate=risk_estimate,
            )
        
        # === Stage 7: Contextual Gate ===
        gate_features = self.gate.extract_features(
            text, pii_evidence, qi_evidence, risk_estimate, self.policy
        )
        gate_features.k_estimate_method = risk_estimate.method if risk_estimate else "unknown"
        gate_decision, gate_reasons = self.gate.decide(gate_features, self.policy)

        # A conservative bound can be zero simply because marginal tables do
        # not establish an intersection; it is not an observed equivalence
        # class of size zero. Let contextual analysis resolve that uncertainty
        # when the gate has marked this specific case unsafe.
        if (
            gate_decision == GateDecision.UNSAFE_ROUTE_LOCAL
            and risk_estimate.method == "conservative_bound"
            and risk_estimate.lower_bound_k is not None
            and risk_estimate.lower_bound_k < self.policy.k_minimum
        ):
            gate_decision = GateDecision.UNCERTAIN_NEED_LLM
        
        contextual_evidence = None
        
        if gate_decision == GateDecision.SAFE_FOR_CLOUD:
            # Fast path: no LLM needed
            tier = Tier.TIER_2_CLOUD_MASKED if pii_evidence else Tier.TIER_1_CLOUD_ORIGINAL
            
        elif gate_decision == GateDecision.UNSAFE_ROUTE_LOCAL:
            # Gate determined unsafe
            tier = Tier.TIER_3_LOCAL
            
        else:
            # === Stage 8: Uncertain - Invoke Local LLM ===
            if self.llm_analyzer:
                start_time = datetime.now()
                contextual_evidence = self.llm_analyzer.analyze(masked_text)
                llm_time = (datetime.now() - start_time).total_seconds()
                
                # Treat abstention/parsing errors as inconclusive rather than hard local-only.
                contextual_unsafe = any((
                    contextual_evidence.public_searchable_event,
                    contextual_evidence.small_community,
                    contextual_evidence.temporal_correlation_risk,
                    contextual_evidence.relationship_network_risk,
                    contextual_evidence.inferential_medical_disclosure,
                    contextual_evidence.rare_combination_indicator,
                    contextual_evidence.unusual_event and
                    contextual_evidence.unusual_event_confidence > 0.7,
                ))
                
                if contextual_unsafe:
                    tier = Tier.TIER_3_LOCAL
                else:
                    tier = Tier.TIER_2_CLOUD_MASKED
            else:
                # LLM disabled, default to masked cloud instead of fail-closed
                tier = Tier.TIER_2_CLOUD_MASKED
                llm_time = 0.0
        
        # === Build Decision Record ===
        uncertainty_flags = []
        if not risk_estimate.estimate_available:
            uncertainty_flags.append("k_estimate_unavailable")
        if risk_estimate.has_unseen_values:
            uncertainty_flags.append("unseen_qi_values")
        if contextual_evidence and contextual_evidence.model_abstained:
            uncertainty_flags.append("llm_abstained")
        
        decision = RoutingDecision(
            tier=tier,
            policy_name=self.policy.name,
            policy_version=self.policy.version,
            hard_stop_reasons=hard_stops,
            uncertainty_flags=uncertainty_flags,
            contextual_review_invoked=contextual_evidence is not None,
            estimated_k_lower=risk_estimate.lower_bound_k,
            high_harm_detected=bool(high_harm),
            direct_identifiers_found=len([
                e for e in pii_evidence 
                if e.category == EntityCategory.DIRECT_IDENTIFIER
            ]),
            qis_found=len(qi_evidence),
            input_hash=input_hash,
            input_length=len(text),
            detector_versions={
                "pii_detector": self.pii_detector.VERSION,
                "qi_detector": self.qi_detector.VERSION,
            }
        )
        
        return RoutingResult(
            decision=decision,
            masked_text=masked_text,
            pii_evidence=pii_evidence,
            qi_evidence=qi_evidence,
            risk_estimate=risk_estimate,
            contextual_evidence=contextual_evidence,
            gate_features=gate_features,
            llm_time=llm_time,
        )
    
    def _apply_masking(self, text: str, pii_evidence: list[PIIEvidence]) -> str:
        """Apply masking to detected PII."""
        if not pii_evidence:
            return text
        
        # Sort by position descending to preserve offsets
        sorted_evidence = sorted(
            pii_evidence, 
            key=lambda e: e.span_start, 
            reverse=True
        )
        
        masked = text
        for e in sorted_evidence:
            # Only mask direct and strong identifiers by default
            if e.category in {EntityCategory.DIRECT_IDENTIFIER, EntityCategory.STRONG_IDENTIFIER}:
                mask_token = f"[{e.entity_type}]"
                masked = masked[:e.span_start] + mask_token + masked[e.span_end:]
                e.masking_action = "masked"
        
        return masked
