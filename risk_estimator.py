"""
risk_estimator.py - Joint risk estimation using local frequency tables

Computes joint k-anonymity estimates with proper uncertainty bounds.
Does NOT multiply marginal probabilities as if independent.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from policy import QIEvidence, JointRiskEstimate, PolicyProfile, HarmCategory
from frequency_tables import LocalFrequencyTable, FrequencyResult


class JointRiskEstimator:
    """
    Estimates joint re-identification risk for QI combinations.
    
    Approach:
    1. Try exact joint lookup first
    2. Fall back to conservative Fréchet bounds if no joint data
    3. Return "unavailable" if any component is missing
    
    Never silently uses independence assumption.
    """
    
    def __init__(self, frequency_table: LocalFrequencyTable):
        self.freq_table = frequency_table
    
    def estimate(
        self,
        qis: list[QIEvidence],
        policy: PolicyProfile
    ) -> JointRiskEstimate:
        """
        Estimate joint equivalence class size for a set of QIs.
        
        Returns conservative lower bound suitable for privacy decisions.
        """
        if not qis:
            return JointRiskEstimate(
                qi_combination=(),
                method="no_qis",
                point_estimate_k=float('inf'),
                lower_bound_k=float('inf'),
                estimate_available=True,
            )
        
        # Build QI value dict for lookup
        # Only include QIs that are present (not negated) and about the patient
        relevant_qis = [
            qi for qi in qis
            if qi.assertion_status == "present" and qi.experiencer == "patient"
        ]
        
        if not relevant_qis:
            return JointRiskEstimate(
                qi_combination=(),
                method="no_patient_qis",
                point_estimate_k=float('inf'),
                lower_bound_k=float('inf'),
                estimate_available=True,
            )
        
        qi_values = {}
        has_unseen = False
        
        for qi in relevant_qis:
            if qi.normalized_value:
                qi_values[qi.qi_type] = qi.normalized_value
                if qi.is_unseen_value:
                    has_unseen = True
        
        if not qi_values:
            return JointRiskEstimate(
                qi_combination=(),
                method="no_normalized_values",
                estimate_available=False,
            )
        
        population_id = policy.reference_population
        pop_info = self.freq_table.get_population_info(population_id)
        
        if not pop_info:
            return JointRiskEstimate(
                qi_combination=tuple(sorted(qi_values.keys())),
                method="unavailable",
                estimate_available=False,
                has_unseen_values=has_unseen,
            )
        
        # Try exact joint lookup
        joint_result = self.freq_table.lookup_joint(population_id, qi_values)
        
        if joint_result.is_available and joint_result.is_exact_match:
            return JointRiskEstimate(
                qi_combination=tuple(sorted(qi_values.keys())),
                method="empirical_joint",
                point_estimate_k=joint_result.count,
                lower_bound_k=(
                    joint_result.lower_bound * pop_info['total_size']
                    if joint_result.lower_bound is not None
                    else joint_result.count
                ),
                upper_bound_k=(
                    joint_result.upper_bound * pop_info['total_size']
                    if joint_result.upper_bound is not None
                    else joint_result.count
                ),
                reference_population=population_id,
                population_size=pop_info['total_size'],
                estimation_confidence=0.9,
                used_independence_assumption=False,
                has_unseen_values=has_unseen,
                estimate_available=True,
            )
        
        # Fall back to conservative bounds
        conservative_result = self.freq_table.estimate_joint_conservative(
            population_id, qi_values
        )
        
        if conservative_result.is_available:
            return JointRiskEstimate(
                qi_combination=tuple(sorted(qi_values.keys())),
                method="conservative_bound",
                point_estimate_k=None,  # No point estimate - only bounds
                lower_bound_k=conservative_result.lower_bound,
                upper_bound_k=conservative_result.upper_bound,
                reference_population=population_id,
                population_size=pop_info['total_size'],
                estimation_confidence=0.5,  # Lower confidence for bounds
                used_independence_assumption=False,  # Fréchet bounds, not independence
                has_unseen_values=has_unseen,
                estimate_available=True,
            )
        
        # Cannot estimate
        return JointRiskEstimate(
            qi_combination=tuple(sorted(qi_values.keys())),
            method="unavailable",
            estimate_available=False,
            has_unseen_values=has_unseen,
        )
    
    def check_high_harm_qis(
        self,
        qis: list[QIEvidence],
        policy: PolicyProfile
    ) -> list[tuple[QIEvidence, HarmCategory]]:
        """
        Check if any QIs fall into high-harm categories.
        
        These should route to Tier 3 regardless of k-anonymity.
        """
        
        high_harm_found = []
        
        # Rare disease check
        rare_disease_terms = {'ehlers-danlos', 'huntington', 'marfan', 'cystic fibrosis'}
        
        for qi in qis:
            if qi.qi_type == "condition" and qi.normalized_value:
                # Check for rare diseases
                for term in rare_disease_terms:
                    if term in qi.normalized_value.lower():
                        high_harm_found.append((qi, HarmCategory.RARE_DISEASE))
                        break
                
                # Check for psychiatric conditions
                psychiatric_terms = {'depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd'}
                for term in psychiatric_terms:
                    if term in qi.normalized_value.lower():
                        if HarmCategory.PSYCHIATRIC in policy.high_harm_categories:
                            high_harm_found.append((qi, HarmCategory.PSYCHIATRIC))
                        break
                
                # Check for substance use
                substance_terms = {'addiction', 'substance use', 'alcoholism', 'opioid'}
                for term in substance_terms:
                    if term in qi.normalized_value.lower():
                        if HarmCategory.SUBSTANCE_USE in policy.high_harm_categories:
                            high_harm_found.append((qi, HarmCategory.SUBSTANCE_USE))
                        break
        
        return high_harm_found
