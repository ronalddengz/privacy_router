#!/usr/bin/env python3
"""
Benchmark script for the revised privacy-preserving text router pipeline.

Tracks:
- Tier distribution (TIER_1, TIER_2, TIER_3)
- False cloud release rate (expected_critical=True routed to cloud)
- Over-censoring rate (expected_critical=False routed to Tier 3)
- LLM invocation rate
- k_lower estimates and distributions
- Processing time metrics
- Multi-k sweep for optimal threshold selection

Usage:
    python benchmark_revised.py example_inputs.txt
    python benchmark_revised.py example_inputs.txt --fresh --max-samples 50
    python benchmark_revised.py example_inputs.txt -o results/ -k 5
"""

import argparse
import csv
import hashlib
import json
import signal
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import re

import numpy as np

# Import the revised pipeline components
from policy import PolicyProfile, Tier, HarmCategory
from frequency_tables import LocalFrequencyTable, create_sample_frequency_table
from router import PrivacyRouter, RoutingResult
from contextual_gate import GateDecision


def _json_safe(value: Any) -> Any:
    """Recursively convert non-finite floats to null so JSON stays standards-compliant."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    return value


def _write_json(path: Path, payload: Any):
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, default=str, allow_nan=False)


def _tier_name(tier_value: int) -> str:
    return {
        Tier.TIER_1_CLOUD_ORIGINAL.value: "TIER_1_CLOUD_ORIGINAL",
        Tier.TIER_2_CLOUD_MASKED.value: "TIER_2_CLOUD_MASKED",
        Tier.TIER_3_LOCAL.value: "TIER_3_LOCAL",
    }.get(tier_value, str(tier_value))


def _write_timing_summary(output_dir: Path, results: List["RevisedPipelineResult"]):
    """Write a per-sample timing/tier summary for quick inspection."""
    rows = []
    for result in results:
        rows.append({
            "sample_name": result.sample_name,
            "expected_critical": result.expected_critical,
            "tier": result.tier,
            "tier_name": _tier_name(result.tier),
            "total_time_seconds": round(result.total_time, 6),
            "masking_ratio": round(result.masking_ratio, 6),
            "k_lower": round(result.k_lower, 6),
        })

    rows.sort(key=lambda row: row["total_time_seconds"], reverse=True)

    json_path = output_dir / "sample_timing_summary.json"
    csv_path = output_dir / "sample_timing_summary.csv"

    _write_json(json_path, rows)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_name",
                "expected_critical",
                "tier",
                "tier_name",
                "total_time_seconds",
                "masking_ratio",
                "k_lower",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-sample timing summary to {json_path}")
    print(f"Saved per-sample timing CSV to {csv_path}")

    if rows:
        print("\nPer-sample timing/tier summary:")
        for row in rows:
            print(
                f"  - {row['sample_name']}: tier={row['tier_name']} | "
                f"time={row['total_time_seconds']:.3f}s | k_lower={row['k_lower']:.3f}"
            )


def _load_results_from_dicts(result_dicts: List[Dict[str, Any]]) -> List["RevisedPipelineResult"]:
    """Rehydrate serialized results, restoring integer keys used by multi-k analysis."""
    results = []

    for data in result_dicts:
        tiers_by_k_raw = data.get("tiers_by_k", {}) or {}
        k_lowers_by_k_raw = data.get("k_lowers_by_k", {}) or {}

        data = dict(data)
        data["tiers_by_k"] = {int(k): v for k, v in tiers_by_k_raw.items()}
        data["k_lowers_by_k"] = {int(k): v for k, v in k_lowers_by_k_raw.items()}
        results.append(RevisedPipelineResult(**data))

    return results


def _has_positive_k_lower(result: "RevisedPipelineResult") -> bool:
    return result.k_lower is not None and result.k_lower > 0


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class BenchmarkSample:
    """A single benchmark sample with expected outcome."""
    name: str
    text: str
    expected_critical: bool  # True = should NOT go to cloud
    
    def get_hash(self) -> str:
        """Generate unique hash for this sample."""
        content = f"{self.name}:{self.text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class RevisedPipelineResult:
    """Result from running the revised pipeline on a single sample."""
    # Sample info
    sample_name: str
    sample_hash: str
    text_length: int
    expected_critical: bool
    
    # Routing decision
    tier: int  # Tier enum value
    gate_decision: str  # SAFE_FOR_CLOUD, UNSAFE_ROUTE_LOCAL, UNCERTAIN_NEED_LLM
    gate_reasons: List[str]
    
    # Risk estimates
    k_lower: float
    k_upper: float
    joint_risk_score: float
    
    # Detection counts
    direct_identifier_count: int
    quasi_identifier_count: int
    high_harm_categories: List[str]
    
    # LLM usage
    llm_invoked: bool
    llm_decision: Optional[str]
    llm_explanation: Optional[str]
    
    # Masking info
    original_text_length: int
    masked_text_length: int
    masking_ratio: float  # (original - masked) / original
    
    # Flags
    hard_stop_triggered: bool
    hard_stop_reasons: List[str]
    uncertainty_flags: List[str]
    
    # Timing
    total_time: float
    detection_time: float
    risk_estimation_time: float
    gate_time: float
    llm_time: float
    
    # Multi-k analysis (results at different k_minimum values)
    tiers_by_k: Dict[int, str] = field(default_factory=dict)
    k_lowers_by_k: Dict[int, float] = field(default_factory=dict)
    
    # Metadata
    processed_at: str = ""


@dataclass
class CheckpointData:
    """Data structure for checkpoint files."""
    input_file_hash: str
    k_minimum: int
    total_samples: int
    processed_count: int
    results: List[Dict]  # Serialized RevisedPipelineResults
    last_updated: str
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointData':
        return cls(**data)


@dataclass
class BenchmarkAnalysis:
    """Aggregate analysis of benchmark results."""
    total_samples: int
    
    # Tier distribution
    tier_1_count: int
    tier_2_count: int
    tier_3_count: int
    tier_1_pct: float
    tier_2_pct: float
    tier_3_pct: float
    
    # Safety metrics
    false_cloud_release_count: int  # expected_critical=True but routed to cloud
    false_cloud_release_rate: float
    over_censoring_count: int  # expected_critical=False but routed to Tier 3
    over_censoring_rate: float
    
    # LLM usage
    llm_invocation_count: int
    llm_invocation_rate: float
    
    # k-anonymity statistics
    mean_k_lower: float
    median_k_lower: float
    mean_k_lower_by_tier: Dict[str, float]
    
    # Timing
    mean_total_time: float
    mean_llm_time: float
    
    # Feature correlations with expected_critical
    correlations: Dict[str, float]
    
    # Multi-k optimal thresholds
    optimal_k_knee: Optional[int]
    optimal_k_efficiency: Optional[int]
    optimal_k_pareto: List[int]


# =============================================================================
# Checkpoint Manager
# =============================================================================
class CheckpointManager:
    """Manages saving and loading of benchmark progress."""
    
    def __init__(self, output_dir: Path, input_file: str, k_minimum: int):
        self.output_dir = output_dir
        self.input_file = input_file
        self.k_minimum = k_minimum
        self.checkpoint_file = output_dir / "checkpoint_revised.json"
        self.backup_file = output_dir / "checkpoint_revised.backup.json"
        
        # Calculate input file hash for validation
        self.input_hash = self._hash_input_file()
        
        # Results storage
        self.results: Dict[str, RevisedPipelineResult] = {}  # hash -> result
        self.processed_hashes: Set[str] = set()
        
        # Interrupt handling
        self._interrupted = False
        self._setup_signal_handlers()
    
    def _hash_input_file(self) -> str:
        """Generate hash of input file for change detection."""
        with open(self.input_file, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _setup_signal_handlers(self):
        """Setup graceful interrupt handling."""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        if not self._interrupted:
            print("\n\n⚠️  Interrupt received - saving checkpoint...")
            self._interrupted = True
            self.save_checkpoint()
            print("✓ Checkpoint saved. Run again to resume.")
    
    @property
    def was_interrupted(self) -> bool:
        return self._interrupted
    
    def is_processed(self, sample: BenchmarkSample) -> bool:
        """Check if a sample has already been processed."""
        return sample.get_hash() in self.processed_hashes
    
    def add_result(self, sample: BenchmarkSample, result: RevisedPipelineResult):
        """Add a result and mark sample as processed."""
        sample_hash = sample.get_hash()
        self.results[sample_hash] = result
        self.processed_hashes.add(sample_hash)
    
    def get_result(self, sample: BenchmarkSample) -> Optional[RevisedPipelineResult]:
        """Get cached result for a sample."""
        return self.results.get(sample.get_hash())
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        # Backup existing checkpoint
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.rename(self.backup_file)
            except Exception:
                pass
        
        checkpoint_data = CheckpointData(
            input_file_hash=self.input_hash,
            k_minimum=self.k_minimum,
            total_samples=len(self.results),
            processed_count=len(self.processed_hashes),
            results=[asdict(r) for r in self.results.values()],
            last_updated=datetime.now().isoformat(),
        )
        
        _write_json(self.checkpoint_file, checkpoint_data.to_dict())
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if valid. Returns True if loaded."""
        if not self.checkpoint_file.exists():
            print("No checkpoint found - starting fresh")
            return False
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            
            checkpoint = CheckpointData.from_dict(data)
            
            # Validate checkpoint
            if checkpoint.input_file_hash != self.input_hash:
                print("Input file changed - invalidating checkpoint")
                return False
            
            if checkpoint.k_minimum != self.k_minimum:
                print(f"k_minimum changed ({checkpoint.k_minimum} -> {self.k_minimum}) - invalidating checkpoint")
                return False
            
            # Restore results
            for result_dict in checkpoint.results:
                result = self._dict_to_result(result_dict)
                self.results[result.sample_hash] = result
                self.processed_hashes.add(result.sample_hash)
            
            print(f"✓ Loaded checkpoint: {len(self.processed_hashes)} samples already processed")
            print(f"  Last updated: {checkpoint.last_updated}")
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False
    
    def _dict_to_result(self, d: Dict) -> RevisedPipelineResult:
        """Convert dict back to RevisedPipelineResult."""
        return _load_results_from_dicts([d])[0]
    
    def clear_checkpoint(self):
        """Clear existing checkpoint."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.backup_file.exists():
            self.backup_file.unlink()
        self.results.clear()
        self.processed_hashes.clear()
        print("✓ Checkpoint cleared")
    
    def get_all_results(self) -> List[RevisedPipelineResult]:
        """Get all results in order."""
        return list(self.results.values())


# =============================================================================
# Input Parser
# =============================================================================
def parse_input_file(filepath: str, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
    """Parse benchmark samples from input file."""
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by --- separator
    blocks = content.split('---')
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Parse metadata comments
        name = None
        expected_critical = False
        text_lines = []
        
        for line in block.split('\n'):
            if line.strip().startswith('# name:'):
                name = line.split(':', 1)[1].strip()
            elif line.strip().startswith('# expected_critical:'):
                value = line.split(':', 1)[1].strip().lower()
                expected_critical = value == 'true'
            elif not line.strip().startswith('#'):
                text_lines.append(line)
        
        text = '\n'.join(text_lines).strip()
        
        if text and name:
            samples.append(BenchmarkSample(
                name=name,
                text=text,
                expected_critical=expected_critical
            ))
        
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples


# =============================================================================
# Pipeline Runner
# =============================================================================
class RevisedBenchmarkPipeline:
    """Runs the revised privacy router pipeline with checkpointing."""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        k_minimum: int = 5,
        db_path: str = "frequency_tables.db"
    ):
        self.checkpoint = checkpoint_manager
        self.k_minimum = k_minimum
        self.db_path = db_path
        
        # Initialize components
        print("Initializing frequency table...")
        self._init_frequency_table()
        
        print("Initializing privacy router...")
        self._init_router()
    
    def _init_frequency_table(self):
        """Initialize or load frequency table."""
        db_path = Path(self.db_path)
        if not db_path.exists():
            print(f"  Creating sample frequency table at {db_path}...")
        else:
            print(f"  Refreshing sample frequency table at {db_path}...")
        create_sample_frequency_table(str(db_path))
        self.freq_table = LocalFrequencyTable(str(db_path))
    
    def _init_router(self):
        """Initialize the privacy router."""
        policy = PolicyProfile(
            name="benchmark_profile",
            version="1.0.0",
            reference_population="hospital_2024",
            k_minimum=self.k_minimum,
            k_safe_threshold=20,
            high_harm_categories={
                HarmCategory.PSYCHIATRIC,
                HarmCategory.SUBSTANCE_USE,
                HarmCategory.REPRODUCTIVE,
                HarmCategory.GENETIC,
                HarmCategory.HIV_STI,
                HarmCategory.RARE_DISEASE,
            },
            cloud_allows_phi=False,
            cloud_allows_masked_phi=True,
            require_joint_estimate=False,
        )
        self.router = PrivacyRouter(
            policy=policy,
            frequency_table=self.freq_table,
        )
    
    def _run_with_k(self, text: str, k: int) -> Tuple[str, float]:
        """Run routing with a specific k_minimum and return (tier, k_lower)."""
        # Create a new policy with the specified k
        policy = PolicyProfile(
            name=self.router.policy.name,
            version=self.router.policy.version,
            reference_population=self.router.policy.reference_population,
            k_minimum=k,
            k_safe_threshold=self.router.policy.k_safe_threshold,
            high_harm_categories=self.router.policy.high_harm_categories,
            cloud_allows_phi=self.router.policy.cloud_allows_phi,
            cloud_allows_masked_phi=self.router.policy.cloud_allows_masked_phi,
            require_joint_estimate=self.router.policy.require_joint_estimate,
        )
        
        temp_router = PrivacyRouter(
            policy=policy,
            frequency_table=self.freq_table,
        )
        
        result = temp_router.route(text)
        return (
            result.decision.tier.value,
            result.risk_estimate.lower_bound_k if result.risk_estimate and result.risk_estimate.lower_bound_k is not None else 0.0,
        )
    
    def run_sample(self, sample: BenchmarkSample) -> RevisedPipelineResult:
        """Run the full pipeline on a single sample."""
        total_start = time.time()
        
        # Run the main pipeline
        result = self.router.route(sample.text)
        
        total_time = time.time() - total_start
        
        # Extract timing components (approximate)
        detection_time = total_time * 0.3  # Rough estimate
        risk_estimation_time = total_time * 0.2
        gate_time = total_time * 0.1
        llm_time = result.llm_time if hasattr(result, 'llm_time') else 0.0
        
        # Multi-k analysis
        k_values = [2, 3, 5, 10, 20, 50, 100]
        tiers_by_k = {}
        k_lowers_by_k = {}
        
        for k in k_values:
            tier_str, k_lower = self._run_with_k(sample.text, k)
            tiers_by_k[k] = tier_str
            k_lowers_by_k[k] = k_lower
        
        # Extract evidence details
        pii_count = len(result.pii_evidence) if result.pii_evidence else 0
        qi_count = len(result.qi_evidence) if result.qi_evidence else 0
        
        high_harm_cats = [
            cat.value
            for _, cat in self.router.risk_estimator.check_high_harm_qis(result.qi_evidence, self.router.policy)
        ]
        
        # Gate info
        if result.decision.contextual_review_invoked:
            gate_decision = "UNCERTAIN_NEED_LLM"
        elif result.decision.tier == Tier.TIER_3_LOCAL:
            gate_decision = "UNSAFE_ROUTE_LOCAL"
        else:
            gate_decision = "SAFE_FOR_CLOUD"
        gate_reasons = result.decision.hard_stop_reasons if result.decision.hard_stop_reasons else []
        
        # LLM info
        llm_invoked = result.decision.contextual_review_invoked
        llm_decision = None
        llm_explanation = None
        
        # Risk estimate
        k_lower = result.risk_estimate.lower_bound_k if result.risk_estimate and result.risk_estimate.lower_bound_k is not None else 0.0
        k_upper = result.risk_estimate.upper_bound_k if result.risk_estimate and result.risk_estimate.upper_bound_k is not None else float('inf')
        joint_risk = 0.0 if not k_lower or k_lower == float('inf') else min(1.0, 1.0 / k_lower)
        
        # Hard stops and uncertainty
        hard_stop = bool(result.decision.hard_stop_reasons)
        hard_stop_reasons = result.decision.hard_stop_reasons
        uncertainty_flags = result.decision.uncertainty_flags
        
        # Masking ratio
        original_len = len(sample.text)
        masked_len = len(result.masked_text) if result.masked_text else original_len
        masking_ratio = (original_len - masked_len) / original_len if original_len > 0 else 0.0
        
        return RevisedPipelineResult(
            sample_name=sample.name,
            sample_hash=sample.get_hash(),
            text_length=len(sample.text),
            expected_critical=sample.expected_critical,
            
            tier=result.decision.tier.value,
            gate_decision=gate_decision,
            gate_reasons=gate_reasons,
            
            k_lower=k_lower,
            k_upper=k_upper,
            joint_risk_score=joint_risk,
            
            direct_identifier_count=pii_count,
            quasi_identifier_count=qi_count,
            high_harm_categories=high_harm_cats,
            
            llm_invoked=llm_invoked,
            llm_decision=llm_decision,
            llm_explanation=llm_explanation,
            
            original_text_length=original_len,
            masked_text_length=masked_len,
            masking_ratio=masking_ratio,
            
            hard_stop_triggered=hard_stop,
            hard_stop_reasons=hard_stop_reasons,
            uncertainty_flags=uncertainty_flags,
            
            total_time=total_time,
            detection_time=detection_time,
            risk_estimation_time=risk_estimation_time,
            gate_time=gate_time,
            llm_time=llm_time,
            
            tiers_by_k=tiers_by_k,
            k_lowers_by_k=k_lowers_by_k,
            
            processed_at=datetime.now().isoformat(),
        )
    
    def run_all(self, samples: List[BenchmarkSample], save_interval: int = 1) -> List[RevisedPipelineResult]:
        """Run pipeline on all samples with checkpointing."""
        total = len(samples)
        processed_new = 0
        skipped = 0
        
        for i, sample in enumerate(samples):
            # Check for interrupt
            if self.checkpoint.was_interrupted:
                print(f"\nStopping at sample {i}/{total} due to interrupt.")
                break
            
            # Check if already processed
            if self.checkpoint.is_processed(sample):
                skipped += 1
                if skipped <= 5 or skipped % 10 == 0:
                    print(f"  [{i+1}/{total}] {sample.name} - CACHED ✓")
                continue
            
            # Process sample
            print(f"  [{i+1}/{total}] Processing: {sample.name}...", end=" ", flush=True)
            
            try:
                start_time = time.time()
                result = self.run_sample(sample)
                elapsed = time.time() - start_time
                
                self.checkpoint.add_result(sample, result)
                processed_new += 1
                
                tier_emoji = {
                    Tier.TIER_1_CLOUD_ORIGINAL.value: "☁️",
                    Tier.TIER_2_CLOUD_MASKED.value: "🔒☁️",
                    Tier.TIER_3_LOCAL.value: "🏠"
                }.get(result.tier, "❓")
                
                tier_name = {
                    Tier.TIER_1_CLOUD_ORIGINAL.value: "TIER_1_CLOUD_ORIGINAL",
                    Tier.TIER_2_CLOUD_MASKED.value: "TIER_2_CLOUD_MASKED",
                    Tier.TIER_3_LOCAL.value: "TIER_3_LOCAL",
                }.get(result.tier, str(result.tier))
                print(f"{tier_emoji} {tier_name} (k_lower={result.k_lower:.1f}) [{elapsed:.2f}s]")
                
                # Save checkpoint periodically
                if processed_new % save_interval == 0:
                    self.checkpoint.save_checkpoint()
                    
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final checkpoint save
        if processed_new > 0:
            self.checkpoint.save_checkpoint()
        
        if skipped > 5:
            print(f"  ... ({skipped} samples loaded from cache)")
        
        print(f"\n✓ Processed {processed_new} new samples, {skipped} from cache")
        
        return self.checkpoint.get_all_results()


# =============================================================================
# Analysis Functions
# =============================================================================
def analyze_results(results: List[RevisedPipelineResult]) -> BenchmarkAnalysis:
    """Analyze benchmark results."""
    if not results:
        return BenchmarkAnalysis(
            total_samples=0,
            tier_1_count=0, tier_2_count=0, tier_3_count=0,
            tier_1_pct=0, tier_2_pct=0, tier_3_pct=0,
            false_cloud_release_count=0, false_cloud_release_rate=0,
            over_censoring_count=0, over_censoring_rate=0,
            llm_invocation_count=0, llm_invocation_rate=0,
            mean_k_lower=0, median_k_lower=0,
            mean_k_lower_by_tier={},
            mean_total_time=0, mean_llm_time=0,
            correlations={},
            optimal_k_knee=None, optimal_k_efficiency=None, optimal_k_pareto=[],
        )
    
    total = len(results)
    
    # Tier distribution
    tier_1 = [r for r in results if r.tier == Tier.TIER_1_CLOUD_ORIGINAL.value]
    tier_2 = [r for r in results if r.tier == Tier.TIER_2_CLOUD_MASKED.value]
    tier_3 = [r for r in results if r.tier == Tier.TIER_3_LOCAL.value]
    
    # Safety metrics
    # False cloud release: expected_critical=True but routed to cloud (Tier 1 or Tier 2)
    false_cloud = [r for r in results if r.expected_critical and r.tier != Tier.TIER_3_LOCAL.value]
    
    # Over-censoring: expected_critical=False but routed to Tier 3
    over_censor = [r for r in results if not r.expected_critical and r.tier == Tier.TIER_3_LOCAL.value]
    
    # LLM usage
    llm_invoked = [r for r in results if r.llm_invoked]
    
    # k-anonymity stats
    k_lowers = [r.k_lower for r in results if _has_positive_k_lower(r)]
    mean_k_lower = np.mean(k_lowers) if k_lowers else 0.0
    median_k_lower = np.median(k_lowers) if k_lowers else 0.0
    
    mean_k_by_tier = {}
    for tier_name, tier_results in [("TIER_1", tier_1), ("TIER_2", tier_2), ("TIER_3", tier_3)]:
        k_vals = [r.k_lower for r in tier_results if _has_positive_k_lower(r)]
        mean_k_by_tier[tier_name] = np.mean(k_vals) if k_vals else 0.0
    
    # Timing
    total_times = [r.total_time for r in results]
    llm_times = [r.llm_time for r in results if r.llm_time > 0]
    
    # Correlations with expected_critical
    def safe_corr(x, y):
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    
    expected = np.array([1.0 if r.expected_critical else 0.0 for r in results])
    
    correlations = {
        "k_lower": safe_corr(expected, np.array([r.k_lower for r in results])),
        "direct_id_count": safe_corr(expected, np.array([r.direct_identifier_count for r in results])),
        "qi_count": safe_corr(expected, np.array([r.quasi_identifier_count for r in results])),
        "high_harm_count": safe_corr(expected, np.array([len(r.high_harm_categories) for r in results])),
        "masking_ratio": safe_corr(expected, np.array([r.masking_ratio for r in results])),
    }
    
    # Multi-k optimal thresholds
    optimal_k_knee = find_optimal_k(results, method='knee')
    optimal_k_efficiency = find_optimal_k(results, method='efficiency')
    optimal_k_pareto = find_optimal_k(results, method='pareto')
    
    return BenchmarkAnalysis(
        total_samples=total,
        tier_1_count=len(tier_1),
        tier_2_count=len(tier_2),
        tier_3_count=len(tier_3),
        tier_1_pct=len(tier_1) / total * 100,
        tier_2_pct=len(tier_2) / total * 100,
        tier_3_pct=len(tier_3) / total * 100,
        false_cloud_release_count=len(false_cloud),
        false_cloud_release_rate=len(false_cloud) / total * 100 if total > 0 else 0,
        over_censoring_count=len(over_censor),
        over_censoring_rate=len(over_censor) / total * 100 if total > 0 else 0,
        llm_invocation_count=len(llm_invoked),
        llm_invocation_rate=len(llm_invoked) / total * 100 if total > 0 else 0,
        mean_k_lower=mean_k_lower,
        median_k_lower=median_k_lower,
        mean_k_lower_by_tier=mean_k_by_tier,
        mean_total_time=np.mean(total_times) if total_times else 0,
        mean_llm_time=np.mean(llm_times) if llm_times else 0,
        correlations=correlations,
        optimal_k_knee=optimal_k_knee,
        optimal_k_efficiency=optimal_k_efficiency,
        optimal_k_pareto=optimal_k_pareto,
    )


def find_optimal_k(results: List[RevisedPipelineResult], method: str = 'knee') -> Optional[int]:
    """Find optimal k_minimum threshold using various methods."""
    if not results or not results[0].tiers_by_k:
        return None
    
    k_values = sorted(results[0].tiers_by_k.keys())
    
    # Calculate metrics at each k
    false_cloud_rates = []
    over_censor_rates = []
    tier_3_rates = []
    
    for k in k_values:
        tier_3_count = sum(1 for r in results if r.tiers_by_k.get(k) == Tier.TIER_3_LOCAL.value)
        
        # False cloud release at this k
        false_cloud = sum(
            1 for r in results 
            if r.expected_critical and r.tiers_by_k.get(k) != Tier.TIER_3_LOCAL.value
        )
        
        # Over-censoring at this k
        over_censor = sum(
            1 for r in results
            if not r.expected_critical and r.tiers_by_k.get(k) == Tier.TIER_3_LOCAL.value
        )
        
        total = len(results)
        false_cloud_rates.append(false_cloud / total if total > 0 else 0)
        over_censor_rates.append(over_censor / total if total > 0 else 0)
        tier_3_rates.append(tier_3_count / total if total > 0 else 0)
    
    if method == 'knee':
        # Find knee point in false_cloud_rate curve (where gains diminish)
        # Use second derivative to find inflection point
        if len(false_cloud_rates) < 3:
            return k_values[0]
        
        # First derivative (rate of change)
        d1 = np.diff(false_cloud_rates)
        # Second derivative (acceleration)
        d2 = np.diff(d1)
        
        # Find max curvature (most negative second derivative)
        if len(d2) > 0:
            knee_idx = np.argmin(d2) + 1  # +1 because d2 is shorter
            return k_values[min(knee_idx, len(k_values) - 1)]
        return k_values[0]
    
    elif method == 'efficiency':
        # Maximize privacy_gain / utility_loss
        # Privacy gain = reduction in false_cloud_rate
        # Utility loss = increase in over_censor_rate
        efficiencies = []
        baseline_fcr = false_cloud_rates[0] if false_cloud_rates else 0
        baseline_ocr = over_censor_rates[0] if over_censor_rates else 0
        
        for i, k in enumerate(k_values):
            privacy_gain = baseline_fcr - false_cloud_rates[i]
            utility_loss = over_censor_rates[i] - baseline_ocr + 0.001  # Avoid div by 0
            efficiencies.append(privacy_gain / utility_loss)
        
        if efficiencies:
            best_idx = np.argmax(efficiencies)
            return k_values[best_idx]
        return k_values[0]
    
    elif method == 'pareto':
        # Find Pareto-optimal k values (minimize false_cloud_rate AND over_censor_rate)
        pareto_optimal = []
        
        for i, k in enumerate(k_values):
            is_dominated = False
            for j, other_k in enumerate(k_values):
                if i == j:
                    continue
                # Check if other_k dominates k
                if (false_cloud_rates[j] <= false_cloud_rates[i] and 
                    over_censor_rates[j] <= over_censor_rates[i] and
                    (false_cloud_rates[j] < false_cloud_rates[i] or 
                     over_censor_rates[j] < over_censor_rates[i])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(k)
        
        return pareto_optimal if pareto_optimal else [k_values[0]]
    
    return None


# =============================================================================
# Visualization Functions
# =============================================================================
def create_visualizations(results: List[RevisedPipelineResult], output_dir: Path):
    """Create visualizations for benchmark results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping visualizations")
        return
    
    if not results:
        print("No results to visualize")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Figure 1: Tier Distribution and Safety Metrics
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Revised Pipeline Benchmark Results', fontsize=14, fontweight='bold')
    
    # Plot 1: Tier distribution pie chart
    ax = axes[0, 0]
    tier_counts = {
        'TIER_1\n(Cloud Original)': sum(1 for r in results if r.tier == Tier.TIER_1_CLOUD_ORIGINAL.value),
        'TIER_2\n(Cloud Masked)': sum(1 for r in results if r.tier == Tier.TIER_2_CLOUD_MASKED.value),
        'TIER_3\n(Local)': sum(1 for r in results if r.tier == Tier.TIER_3_LOCAL.value),
    }
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    wedges, texts, autotexts = ax.pie(
        tier_counts.values(), 
        labels=tier_counts.keys(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title('Tier Distribution')
    
    # Plot 2: Safety metrics bar chart
    ax = axes[0, 1]
    
    total = len(results)
    critical_count = sum(1 for r in results if r.expected_critical)
    non_critical_count = total - critical_count
    
    false_cloud = sum(1 for r in results if r.expected_critical and r.tier != Tier.TIER_3_LOCAL.value)
    over_censor = sum(1 for r in results if not r.expected_critical and r.tier == Tier.TIER_3_LOCAL.value)
    
    metrics = {
        'False Cloud\nRelease Rate': false_cloud / critical_count * 100 if critical_count > 0 else 0,
        'Over-Censoring\nRate': over_censor / non_critical_count * 100 if non_critical_count > 0 else 0,
        'LLM\nInvocation Rate': sum(1 for r in results if r.llm_invoked) / total * 100,
    }
    
    bars = ax.bar(metrics.keys(), metrics.values(), color=['#e74c3c', '#f39c12', '#3498db'])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Safety and Efficiency Metrics')
    ax.set_ylim(0, 100)
    
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom')
    
    # Plot 3: k_lower distribution by tier
    ax = axes[1, 0]
    
    tier_k_lowers = {
        'TIER_1': [r.k_lower for r in results if r.tier == Tier.TIER_1_CLOUD_ORIGINAL.value and _has_positive_k_lower(r)],
        'TIER_2': [r.k_lower for r in results if r.tier == Tier.TIER_2_CLOUD_MASKED.value and _has_positive_k_lower(r)],
        'TIER_3': [r.k_lower for r in results if r.tier == Tier.TIER_3_LOCAL.value and _has_positive_k_lower(r)],
    }
    
    data_to_plot = [tier_k_lowers[t] for t in ['TIER_1', 'TIER_2', 'TIER_3'] if tier_k_lowers[t]]
    labels_to_plot = [t for t in ['TIER_1', 'TIER_2', 'TIER_3'] if tier_k_lowers[t]]
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#f39c12', '#e74c3c'][:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax.set_ylabel('k_lower (log scale)')
    ax.set_yscale('log')
    ax.set_title('k-Anonymity Lower Bound by Tier')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Confusion matrix style heatmap
    ax = axes[1, 1]
    
    # Create 2x3 matrix: expected_critical (T/F) x tier (1/2/3)
    confusion = np.zeros((2, 3))
    for r in results:
        row = 0 if r.expected_critical else 1
        col = {Tier.TIER_1_CLOUD_ORIGINAL.value: 0, Tier.TIER_2_CLOUD_MASKED.value: 1, Tier.TIER_3_LOCAL.value: 2}.get(r.tier, 2)
        confusion[row, col] += 1
    
    im = ax.imshow(confusion, cmap='YlOrRd', aspect='auto')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Tier 1', 'Tier 2', 'Tier 3'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Critical\n(Should Block)', 'Non-Critical\n(Can Release)'])
    ax.set_title('Routing Decisions vs Expected Outcome')
    
    # Add text annotations
    for i in range(2):
        for j in range(3):
            color = 'white' if confusion[i, j] > confusion.max() / 2 else 'black'
            ax.text(j, i, f'{int(confusion[i, j])}', ha='center', va='center', color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'benchmark_overview.png'}")
    
    # =========================================================================
    # Figure 2: Multi-k Analysis
    # =========================================================================
    create_multi_k_visualization(results, output_dir)


def create_multi_k_visualization(results: List[RevisedPipelineResult], output_dir: Path):
    """Create multi-k threshold analysis visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    if not results or not results[0].tiers_by_k:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    k_values = sorted(results[0].tiers_by_k.keys())
    
    # Calculate metrics at each k
    false_cloud_rates = []
    over_censor_rates = []
    tier_3_rates = []
    
    for k in k_values:
        tier_3_count = sum(1 for r in results if r.tiers_by_k.get(k) == Tier.TIER_3_LOCAL.value)
        
        critical = [r for r in results if r.expected_critical]
        non_critical = [r for r in results if not r.expected_critical]
        
        false_cloud = sum(1 for r in critical if r.tiers_by_k.get(k) != Tier.TIER_3_LOCAL.value)
        over_censor = sum(1 for r in non_critical if r.tiers_by_k.get(k) == Tier.TIER_3_LOCAL.value)
        
        false_cloud_rates.append(false_cloud / len(critical) * 100 if critical else 0)
        over_censor_rates.append(over_censor / len(non_critical) * 100 if non_critical else 0)
        tier_3_rates.append(tier_3_count / len(results) * 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-k Threshold Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Rates vs k
    ax = axes[0, 0]
    ax.plot(k_values, false_cloud_rates, 'r-o', label='False Cloud Release Rate', linewidth=2)
    ax.plot(k_values, over_censor_rates, 'b-s', label='Over-Censoring Rate', linewidth=2)
    ax.plot(k_values, tier_3_rates, 'g-^', label='Tier 3 Rate', linewidth=2, alpha=0.7)
    ax.set_xlabel('k_minimum threshold')
    ax.set_ylabel('Rate (%)')
    ax.set_xscale('log')
    ax.set_title('Safety/Utility Trade-off vs k')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pareto frontier
    ax = axes[0, 1]
    ax.scatter(over_censor_rates, false_cloud_rates, c=np.log10(k_values), cmap='viridis', s=100)
    ax.plot(over_censor_rates, false_cloud_rates, 'k--', alpha=0.3)
    
    for i, k in enumerate(k_values):
        ax.annotate(f'k={k}', (over_censor_rates[i], false_cloud_rates[i]), 
                   fontsize=8, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Over-Censoring Rate (%) - Utility Loss')
    ax.set_ylabel('False Cloud Release Rate (%) - Privacy Loss')
    ax.set_title('Privacy-Utility Pareto Frontier')
    ax.grid(True, alpha=0.3)
    
    # Add ideal point
    ax.plot(0, 0, 'g*', markersize=15, label='Ideal (0,0)')
    ax.legend()
    
    # Plot 3: Efficiency curve
    ax = axes[1, 0]
    
    # Efficiency = privacy_gain / utility_loss
    baseline_fcr = false_cloud_rates[0]
    baseline_ocr = over_censor_rates[0]
    
    efficiencies = []
    for i in range(len(k_values)):
        privacy_gain = baseline_fcr - false_cloud_rates[i]
        utility_loss = over_censor_rates[i] - baseline_ocr + 0.1  # Small epsilon
        efficiencies.append(privacy_gain / utility_loss if utility_loss > 0 else 0)
    
    ax.plot(k_values, efficiencies, 'purple', marker='D', linewidth=2)
    
    if efficiencies:
        best_idx = np.argmax(efficiencies)
        ax.axvline(k_values[best_idx], color='purple', linestyle='--', alpha=0.5, 
                  label=f'Best k={k_values[best_idx]}')
        ax.scatter([k_values[best_idx]], [efficiencies[best_idx]], color='purple', s=200, marker='*', zorder=5)
    
    ax.set_xlabel('k_minimum threshold')
    ax.set_ylabel('Efficiency (Privacy Gain / Utility Loss)')
    ax.set_xscale('log')
    ax.set_title('Efficiency Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Tier distribution stacked area
    ax = axes[1, 1]
    
    tier_1_rates = []
    tier_2_rates = []
    tier_3_rates_area = []
    
    for k in k_values:
        t1 = sum(1 for r in results if r.tiers_by_k.get(k) == Tier.TIER_1_CLOUD_ORIGINAL.value) / len(results) * 100
        t2 = sum(1 for r in results if r.tiers_by_k.get(k) == Tier.TIER_2_CLOUD_MASKED.value) / len(results) * 100
        t3 = sum(1 for r in results if r.tiers_by_k.get(k) == Tier.TIER_3_LOCAL.value) / len(results) * 100
        tier_1_rates.append(t1)
        tier_2_rates.append(t2)
        tier_3_rates_area.append(t3)
    
    ax.stackplot(k_values, tier_1_rates, tier_2_rates, tier_3_rates_area,
                 labels=['Tier 1 (Cloud)', 'Tier 2 (Masked)', 'Tier 3 (Local)'],
                 colors=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
    ax.set_xlabel('k_minimum threshold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xscale('log')
    ax.set_title('Tier Distribution vs k')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_k_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'multi_k_analysis.png'}")


# =============================================================================
# Main Benchmark Runner
# =============================================================================
def run_benchmark(
    input_file: str,
    output_dir: str = "benchmark_revised_outputs",
    max_samples: Optional[int] = None,
    k_minimum: int = 5,
    fresh_start: bool = False,
    db_path: str = "frequency_tables.db",
):
    """
    Run the benchmark pipeline.
    
    Args:
        input_file: Path to input samples file
        output_dir: Output directory for results
        max_samples: Limit number of samples (None = all)
        k_minimum: k-anonymity threshold for routing decisions
        fresh_start: If True, ignore existing checkpoint
        db_path: Path to frequency table database
    """
    print("=" * 70)
    print("REVISED PRIVACY ROUTER BENCHMARK")
    print("=" * 70)
    print()
    
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print(f"Loading samples from: {input_file}")
    samples = parse_input_file(input_file, max_samples)
    print(f"Loaded {len(samples)} benchmark samples")
    
    critical_count = sum(1 for s in samples if s.expected_critical)
    print(f"  Expected critical: {critical_count}")
    print(f"  Expected non-critical: {len(samples) - critical_count}")
    
    if max_samples:
        print(f"(Limited to {max_samples} samples)")
    print()
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager(out_path, input_file, k_minimum)
    
    # Handle fresh start
    if fresh_start:
        print("Fresh start requested - clearing checkpoint...")
        checkpoint.clear_checkpoint()
    else:
        checkpoint.load_checkpoint()
    
    # Initialize pipeline with checkpoint
    pipeline = RevisedBenchmarkPipeline(
        checkpoint_manager=checkpoint,
        k_minimum=k_minimum,
        db_path=db_path,
    )
    
    print("\n" + "-" * 70)
    print("PROCESSING SAMPLES")
    print("-" * 70)
    
    results = pipeline.run_all(samples)
    
    # Save raw results
    results_data = [asdict(r) for r in results]
    _write_json(out_path / "pipeline_results.json", results_data)
    print(f"\nSaved pipeline results to {out_path / 'pipeline_results.json'}")

    _write_timing_summary(out_path, results)
    
    # Only generate analysis if we have results and weren't interrupted
    if results and not checkpoint.was_interrupted:
        print("\nGenerating visualizations...")
        create_visualizations(results, out_path)
        
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        
        analysis = analyze_results(results)
        
        print(f"\nTotal Samples: {analysis.total_samples}")
        
        print(f"\nTier Distribution:")
        print(f"  Tier 1 (Cloud Original): {analysis.tier_1_count} ({analysis.tier_1_pct:.1f}%)")
        print(f"  Tier 2 (Cloud Masked):   {analysis.tier_2_count} ({analysis.tier_2_pct:.1f}%)")
        print(f"  Tier 3 (Local Only):     {analysis.tier_3_count} ({analysis.tier_3_pct:.1f}%)")
        
        print(f"\nSafety Metrics:")
        print(f"  False Cloud Release Rate: {analysis.false_cloud_release_rate:.2f}%")
        print(f"    ({analysis.false_cloud_release_count} critical samples released to cloud)")
        print(f"  Over-Censoring Rate: {analysis.over_censoring_rate:.2f}%")
        print(f"    ({analysis.over_censoring_count} non-critical samples routed to local)")
        
        print(f"\nEfficiency Metrics:")
        print(f"  LLM Invocation Rate: {analysis.llm_invocation_rate:.2f}%")
        print(f"    ({analysis.llm_invocation_count} samples required LLM)")
        
        print(f"\nk-Anonymity Statistics:")
        print(f"  Mean k_lower: {analysis.mean_k_lower:.2f}")
        print(f"  Median k_lower: {analysis.median_k_lower:.2f}")
        print(f"  Mean k_lower by tier:")
        for tier, mean_k in analysis.mean_k_lower_by_tier.items():
            print(f"    {tier}: {mean_k:.2f}")
        
        print(f"\nTiming:")
        print(f"  Mean total time: {analysis.mean_total_time:.3f}s")
        print(f"  Mean LLM time: {analysis.mean_llm_time:.3f}s")
        
        print(f"\nFeature Correlations with expected_critical:")
        for feature, corr in sorted(analysis.correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature}: {corr:+.3f}")
        
        print(f"\nOptimal k Thresholds:")
        print(f"  Knee method: k={analysis.optimal_k_knee}")
        print(f"  Efficiency method: k={analysis.optimal_k_efficiency}")
        print(f"  Pareto optimal: k∈{analysis.optimal_k_pareto}")
        
        # Save analysis
        analysis_dict = asdict(analysis)
        _write_json(out_path / "analysis.json", analysis_dict)
        print(f"\nSaved analysis to {out_path / 'analysis.json'}")
        
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
    
    elif checkpoint.was_interrupted:
        print("\n⚠️  Benchmark interrupted - run again to resume")
    
    return results


def load_results_from_output_dir(output_dir: Path) -> List[RevisedPipelineResult]:
    """Load benchmark results from a saved pipeline_results.json file."""
    results_file = output_dir / "pipeline_results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"No saved results found at {results_file}")

    with open(results_file, "r", encoding="utf-8") as f:
        result_dicts = json.load(f)

    if not isinstance(result_dicts, list):
        raise ValueError(f"Expected a list of results in {results_file}")

    return _load_results_from_dicts(result_dicts)


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark revised privacy router pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark (will resume from checkpoint if exists)
  python benchmark_revised.py example_inputs.txt

  # Force fresh start (ignore checkpoint)
  python benchmark_revised.py example_inputs.txt --fresh

  # Limit to first 50 samples
  python benchmark_revised.py example_inputs.txt -n 50

  # Custom k_minimum threshold
  python benchmark_revised.py example_inputs.txt -k 10

Checkpoint Behavior:
  - Progress is saved after each sample
  - If interrupted (Ctrl+C), run again to resume
  - Checkpoint invalidated if input file or k_minimum changes
  - Use --fresh to force restart
        """
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input text file containing benchmark samples"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="benchmark_revised_outputs",
        help="Directory to save output files (default: benchmark_revised_outputs)"
    )
    parser.add_argument(
        "-n", "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "-k", "--k-minimum",
        type=int,
        default=5,
        help="k-anonymity minimum threshold (default: 5)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="frequency_tables.db",
        help="Path to frequency table database (default: frequency_tables.db)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and start fresh"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmark execution and regenerate plots from saved pipeline_results.json"
    )
    
    args = parser.parse_args()
    
    if args.plot_only:
        output_dir = Path(args.output_dir)
        results = load_results_from_output_dir(output_dir)
        print(f"Loaded {len(results)} saved results from {output_dir / 'pipeline_results.json'}")
        print("Generating visualizations from saved results...")
        create_visualizations(results, output_dir)
    else:
        run_benchmark(
            input_file=args.input_file,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            k_minimum=args.k_minimum,
            db_path=args.db_path,
            fresh_start=args.fresh,
        )