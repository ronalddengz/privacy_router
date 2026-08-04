"""
example_usage.py - Demonstrates the privacy router
"""

from pathlib import Path
from policy import QIEvidence, JointRiskEstimate, PolicyProfile, HarmCategory, Tier
from frequency_tables import create_sample_frequency_table, LocalFrequencyTable
from router import PrivacyRouter


def main():
    # === Setup ===
    
    # Create policy profile
    policy = PolicyProfile(
        name="hospital_standard",
        version="1.0.0",
        reference_population="hospital_2024",
        k_minimum=5,
        k_safe_threshold=20,
        high_harm_categories=frozenset({
            HarmCategory.PSYCHIATRIC,
            HarmCategory.SUBSTANCE_USE,
            HarmCategory.REPRODUCTIVE,
            HarmCategory.RARE_DISEASE,
        }),
        require_joint_estimate=True,
        cloud_allows_phi=False,
        cloud_allows_masked_phi=True,
    )
    
    # Initialize frequency table
    db_path = Path("./frequency_data.db")
    freq_table = create_sample_frequency_table(db_path)
    
    # Create router
    router = PrivacyRouter(
        policy=policy,
        frequency_table=freq_table,
        enable_contextual_llm=True
    )
    
    # === Test Cases ===
    
    test_cases = [
        # Case 1: Low risk - should route to cloud
        {
            "name": "Low Risk Clinical Note",
            "text": "Patient presents with common cold symptoms. Recommending rest and fluids.",
        },
        
        # Case 2: Has PII but can be masked
        {
            "name": "PII That Can Be Masked",
            "text": "John Smith, DOB 05/15/1980, presents with hypertension. Contact: 555-123-4567.",
        },
        
        # Case 3: High-harm category (rare disease)
        {
            "name": "Rare Disease - High Harm",
            "text": "Patient is a 45 year old male with Ehlers-Danlos syndrome, hypermobility type.",
        },
        
        # Case 4: Low k-anonymity
        {
            "name": "Low k - Unique Combination", 
            "text": "Patient is a 45 year old male zoologist living in rural Montana with Huntington's disease.",
        },
        
        # Case 5: Contextual risk - unique event
        {
            "name": "Unique Public Event",
            "text": "The patient was the sole survivor of the warehouse fire on Industrial Boulevard last week.",
        },
    ]
    
    print("=" * 70)
    print("Privacy Router Demo")
    print("=" * 70)
    
    for case in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Test: {case['name']}")
        print(f"Input: {case['text'][:100]}...")
        print("-" * 70)
        
        result = router.route(case["text"])
        
        print(f"\nRouting Decision: {result.decision.tier.name}")
        print(f"  Hard stops: {result.decision.hard_stop_reasons or 'None'}")
        print(f"  Uncertainty flags: {result.decision.uncertainty_flags or 'None'}")
        print(f"  Contextual LLM invoked: {result.decision.contextual_review_invoked}")
        
        if result.risk_estimate:
            print(f"\nRisk Estimate:")
            print(f"  Method: {result.risk_estimate.method}")
            print(f"  k_lower: {result.risk_estimate.lower_bound_k}")
            print(f"  Available: {result.risk_estimate.estimate_available}")
        
        print(f"\nMasked text: {result.masked_text[:100]}...")
        
        # Show what would happen
        if result.decision.tier == Tier.TIER_1_CLOUD_ORIGINAL:
            print("\n→ Action: Send original text to cloud")
        elif result.decision.tier == Tier.TIER_2_CLOUD_MASKED:
            print("\n→ Action: Send masked text to cloud")
        else:
            print("\n→ Action: Process locally only")


if __name__ == "__main__":
    main()