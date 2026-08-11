#!/usr/bin/env python3
"""
benchmark.py

Benchmark script for the revised privacy-preserving text router pipeline.

Tracks:
- Tier distribution
- False cloud release rate
- Over-censoring rate
- LLM invocation rate
- k_lower estimates
- Processing time metrics
- Precision-recall analysis for threshold and weight sweeps

Added sweep experiments:
- tau_review: 0.01 to 0.50 step 0.02
- tau_block:  0.50 to 0.95 step 0.05
- k_minimum:  2 to 20 step 1
- w_direct:   0.1 to 1.0 step 0.1
- w_quasi:    0.1 to 1.0 step 0.1

Usage:
    python benchmark.py example_inputs.txt
    python benchmark.py example_inputs.txt --fresh --max-samples 50
    python benchmark.py example_inputs.txt -o results/ -k 5
    python benchmark.py example_inputs.txt --plot-only -o results/
"""

import argparse
import csv
import hashlib
import json
import signal
import time
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from policy import PolicyProfile, Tier, EntityCategory
from frequency_tables import LocalFrequencyTable
from router import PrivacyRouter, RoutingResult
from contextual_gate import GateDecision


# =============================================================================
# JSON helpers
# =============================================================================

def _write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _frange(start: float, stop: float, step: float) -> List[float]:
    values = []
    x = start
    while x <= stop + step / 10.0:
        values.append(round(x, 10))
        x += step
    return values


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkSample:
    """A single benchmark sample with expected outcome."""
    name: str
    text: str
    expected_critical: bool

    def get_hash(self) -> str:
        content = f"{self.name}:{self.text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


@dataclass
class RevisedPipelineResult:
    """Result from running the revised pipeline on a single sample."""

    # Sample info
    sample_name: str
    sample_hash: str
    text_length: int
    expected_critical: bool

    # Routing decision
    tier: int
    gate_decision: str
    gate_reasons: List[str]

    # Risk estimates
    k_lower: float
    k_upper: float
    joint_risk_score: float

    # Detection counts
    direct_identifier_count: int
    quasi_identifier_count: int

    # LLM usage
    llm_invoked: bool
    llm_decision: Optional[str]
    llm_explanation: Optional[str]

    # Masking info
    original_text_length: int
    masked_text_length: int
    masking_ratio: float

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

    # Multi-k analysis
    tiers_by_k: Dict[int, int] = field(default_factory=dict)
    k_lowers_by_k: Dict[int, float] = field(default_factory=dict)

    # Metadata
    processed_at: str = ""


@dataclass
class BenchmarkAnalysis:
    total_samples: int

    tier_1_count: int
    tier_2_count: int
    tier_3_count: int

    tier_1_pct: float
    tier_2_pct: float
    tier_3_pct: float

    false_cloud_release_count: int
    false_cloud_release_rate: float

    over_censoring_count: int
    over_censoring_rate: float

    llm_invocation_count: int
    llm_invocation_rate: float

    mean_k_lower: float
    median_k_lower: float
    mean_k_lower_by_tier: Dict[str, float]

    mean_total_time: float
    mean_llm_time: float

    correlations: Dict[str, float]


@dataclass
class ThresholdSweepPoint:
    """One operating point in a threshold/weight sweep."""
    sweep_name: str

    tau_review: float
    tau_block: float
    k_minimum: int
    w_direct: float
    w_quasi: float

    precision: float
    recall: float
    f1: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    marked_sensitive_rate: float
    llm_review_rate: float
    block_rate: float


# =============================================================================
# Input parsing
# =============================================================================

def parse_input_file(input_file: str, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
    """
    Parse benchmark input file.

    Expected format is compatible with the merged transcript format:

        # name: sample_name
        # expected_critical: true/false
        [START OF TRANSCRIPT]
        ...
        [END OF TRANSCRIPT]

    Falls back to treating blank-line-separated chunks as non-critical samples.
    """
    path = Path(input_file)
    text = path.read_text(encoding="utf-8")

    samples: List[BenchmarkSample] = []

    pattern = re.compile(
        r"# name:\s*(?P<name>.*?)\n"
        r".*?"
        r"# expected_critical:\s*(?P<critical>true|false)"
        r".*?"
        r"\[START OF TRANSCRIPT\]\s*"
        r"(?P<text>.*?)"
        r"\s*\[END OF TRANSCRIPT\]",
        re.DOTALL | re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        name = match.group("name").strip()
        expected_critical = match.group("critical").strip().lower() == "true"
        sample_text = match.group("text").strip()

        samples.append(
            BenchmarkSample(
                name=name,
                text=sample_text,
                expected_critical=expected_critical,
            )
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    if samples:
        return samples

    # Fallback parser.
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    for i, chunk in enumerate(chunks):
        samples.append(
            BenchmarkSample(
                name=f"sample_{i + 1}",
                text=chunk,
                expected_critical=False,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manages saving and loading benchmark progress."""

    def __init__(self, output_dir: Path, input_file: str, k_minimum: int):
        self.output_dir = output_dir
        self.input_file = input_file
        self.k_minimum = k_minimum

        self.checkpoint_file = output_dir / "checkpoint_revised.json"
        self.backup_file = output_dir / "checkpoint_revised.backup.json"

        self.input_hash = self._hash_input_file()

        self.results: Dict[str, RevisedPipelineResult] = {}
        self.processed_hashes: Set[str] = set()

        self._interrupted = False
        self._setup_signal_handlers()

    @property
    def was_interrupted(self) -> bool:
        return self._interrupted

    def _hash_input_file(self) -> str:
        with open(self.input_file, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        print("\nInterrupt received. Saving checkpoint...")
        self._interrupted = True
        self.save_checkpoint()

    def clear_checkpoint(self):
        for path in [self.checkpoint_file, self.backup_file]:
            if path.exists():
                path.unlink()
        self.results.clear()
        self.processed_hashes.clear()

    def load_checkpoint(self):
        if not self.checkpoint_file.exists():
            return

        try:
            data = _read_json(self.checkpoint_file)

            if data.get("input_hash") != self.input_hash:
                print("Checkpoint input hash mismatch. Ignoring checkpoint.")
                return

            if data.get("k_minimum") != self.k_minimum:
                print("Checkpoint k_minimum mismatch. Ignoring checkpoint.")
                return

            raw_results = data.get("results", {})

            for sample_hash, raw in raw_results.items():
                raw = dict(raw)

                raw["tiers_by_k"] = {
                    int(k): v for k, v in raw.get("tiers_by_k", {}).items()
                }
                raw["k_lowers_by_k"] = {
                    int(k): v for k, v in raw.get("k_lowers_by_k", {}).items()
                }

                self.results[sample_hash] = RevisedPipelineResult(**raw)
                self.processed_hashes.add(sample_hash)

            print(f"Loaded checkpoint with {len(self.results)} processed samples.")

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    def save_checkpoint(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "input_file": self.input_file,
            "input_hash": self.input_hash,
            "k_minimum": self.k_minimum,
            "saved_at": datetime.now().isoformat(),
            "results": {
                h: asdict(r)
                for h, r in self.results.items()
            },
        }

        if self.checkpoint_file.exists():
            self.checkpoint_file.replace(self.backup_file)

        _write_json(self.checkpoint_file, data)

    def add_result(self, sample_hash: str, result: RevisedPipelineResult):
        self.results[sample_hash] = result
        self.processed_hashes.add(sample_hash)

    def is_processed(self, sample_hash: str) -> bool:
        return sample_hash in self.processed_hashes

    def get_result(self, sample_hash: str) -> Optional[RevisedPipelineResult]:
        return self.results.get(sample_hash)


# =============================================================================
# Pipeline Runner
# =============================================================================

class RevisedBenchmarkPipeline:
    """Runs the revised privacy router pipeline with checkpointing."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        k_minimum: int = 5,
        db_path: str = "frequency_tables.db",
    ):
        self.checkpoint = checkpoint_manager
        self.k_minimum = k_minimum
        self.db_path = db_path

        print("Initializing frequency table...")
        self._init_frequency_table()

        print("Initializing privacy router...")
        self._init_router()

    def _init_frequency_table(self):
        db_path = Path(self.db_path)
        if not db_path.exists():
            print(f"  Creating sample frequency table at {db_path}...")
        else:
            print(f"  Refreshing sample frequency table at {db_path}...")

        self.freq_table = LocalFrequencyTable(str(db_path))

    def _init_router(self):
        policy = PolicyProfile(
            name="benchmark_profile",
            version="1.0.0",
            reference_population="hospital_2024",
            k_minimum=self.k_minimum,
            k_safe_threshold=20,
            cloud_allows_phi=False,
            cloud_allows_masked_phi=True,
            require_joint_estimate=True,
            contextual_review_threshold=0.30,
        )

        self.router = PrivacyRouter(
            policy=policy,
            frequency_table=self.freq_table,
        )

    def _run_single_k(self, sample: BenchmarkSample, k_value: int) -> Tuple[int, float]:
        """
        Run router for one alternate k value.

        Used only for multi-k summary. This can be expensive, but keeps the
        saved `tiers_by_k` fields compatible with your existing outputs.
        """
        policy = PolicyProfile(
            name="benchmark_profile",
            version="1.0.0",
            reference_population="hospital_2024",
            k_minimum=k_value,
            k_safe_threshold=20,
            cloud_allows_phi=False,
            cloud_allows_masked_phi=True,
            require_joint_estimate=True,
            contextual_review_threshold=0.30,
        )

        router = PrivacyRouter(
            policy=policy,
            freq_table=self.freq_table,
        )

        result = router.route(sample.text)
        k_lower = (
            result.risk_estimate.lower_bound_k
            if result.risk_estimate and result.risk_estimate.lower_bound_k is not None
            else 0.0
        )

        return result.decision.tier.value, k_lower

    def process_sample(self, sample: BenchmarkSample) -> RevisedPipelineResult:
        start = time.time()

        result: RoutingResult = self.router.route(sample.text)

        total_time = time.time() - start

        # The router currently returns total llm_time, but not all stage timings.
        # Keep approximate fields for backwards compatibility with your plots.
        llm_time = result.llm_time if result.llm_time is not None else 0.0
        non_llm_time = max(total_time - llm_time, 0.0)

        detection_time = non_llm_time * 0.40
        risk_estimation_time = non_llm_time * 0.30
        gate_time = non_llm_time * 0.30

        pii_count = len(result.pii_evidence or [])
        qi_count = len(result.qi_evidence or [])

        direct_identifier_count = len([
            e for e in result.pii_evidence or []
            if e.category == EntityCategory.DIRECT_IDENTIFIER
        ])

        # If your earlier benchmark treated all PII as direct count, use the
        # dedicated direct_identifier_count above but fall back to pii_count.
        if direct_identifier_count == 0:
            direct_identifier_count = pii_count

        if result.decision.tier == Tier.TIER_3_LOCAL:
            if result.decision.contextual_review_invoked:
                gate_decision = "UNCERTAIN_NEED_LLM"
            else:
                gate_decision = "UNSAFE_ROUTE_LOCAL"
        else:
            gate_decision = "SAFE_FOR_CLOUD"

        gate_reasons = (
            result.decision.hard_stop_reasons
            if result.decision.hard_stop_reasons
            else []
        )

        k_lower = (
            result.risk_estimate.lower_bound_k
            if result.risk_estimate and result.risk_estimate.lower_bound_k is not None
            else 0.0
        )
        k_upper = (
            result.risk_estimate.upper_bound_k
            if result.risk_estimate and result.risk_estimate.upper_bound_k is not None
            else float("inf")
        )

        joint_risk = (
            0.0
            if not k_lower or k_lower == float("inf")
            else min(1.0, 1.0 / k_lower)
        )

        original_len = len(sample.text)
        masked_len = len(result.masked_text) if result.masked_text else original_len
        masking_ratio = (
            (original_len - masked_len) / original_len
            if original_len > 0
            else 0.0
        )

        tiers_by_k: Dict[int, int] = {}
        k_lowers_by_k: Dict[int, float] = {}

        for k in [2, 5, 10, 15, 20, 50, 100, 1000, 10000]:
            if k == self.k_minimum:
                tiers_by_k[k] = result.decision.tier.value
                k_lowers_by_k[k] = k_lower
            else:
                try:
                    tier_value, k_value_lower = self._run_single_k(sample, k)
                    tiers_by_k[k] = tier_value
                    k_lowers_by_k[k] = k_value_lower
                except Exception:
                    tiers_by_k[k] = result.decision.tier.value
                    k_lowers_by_k[k] = k_lower

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

            direct_identifier_count=direct_identifier_count,
            quasi_identifier_count=qi_count,

            llm_invoked=result.decision.contextual_review_invoked,
            llm_decision=None,
            llm_explanation=None,

            original_text_length=original_len,
            masked_text_length=masked_len,
            masking_ratio=masking_ratio,

            hard_stop_triggered=bool(result.decision.hard_stop_reasons),
            hard_stop_reasons=result.decision.hard_stop_reasons,
            uncertainty_flags=result.decision.uncertainty_flags,

            total_time=total_time,
            detection_time=detection_time,
            risk_estimation_time=risk_estimation_time,
            gate_time=gate_time,
            llm_time=llm_time,

            tiers_by_k=tiers_by_k,
            k_lowers_by_k=k_lowers_by_k,

            processed_at=datetime.now().isoformat(),
        )

    def run_all(
        self,
        samples: List[BenchmarkSample],
        save_interval: int = 1,
    ) -> List[RevisedPipelineResult]:
        total = len(samples)
        processed_new = 0
        skipped = 0

        for i, sample in enumerate(samples):
            sample_hash = sample.get_hash()

            if self.checkpoint.is_processed(sample_hash):
                skipped += 1
                continue

            print(f"[{i + 1}/{total}] Processing {sample.name}")

            try:
                result = self.process_sample(sample)
                self.checkpoint.add_result(sample_hash, result)
                processed_new += 1

                if processed_new % save_interval == 0:
                    self.checkpoint.save_checkpoint()

            except Exception as e:
                print(f"  ERROR processing {sample.name}: {e}")

            if self.checkpoint.was_interrupted:
                break

        self.checkpoint.save_checkpoint()

        print(f"Processed new: {processed_new}")
        print(f"Skipped existing: {skipped}")

        ordered_results = []
        for sample in samples:
            r = self.checkpoint.get_result(sample.get_hash())
            if r is not None:
                ordered_results.append(r)

        return ordered_results


# =============================================================================
# Analysis
# =============================================================================

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def analyze_results(results: List[RevisedPipelineResult]) -> BenchmarkAnalysis:
    total = len(results)

    tier_1 = [r for r in results if r.tier == Tier.TIER_1_CLOUD_ORIGINAL.value]
    tier_2 = [r for r in results if r.tier == Tier.TIER_2_CLOUD_MASKED.value]
    tier_3 = [r for r in results if r.tier == Tier.TIER_3_LOCAL.value]

    false_cloud = [
        r for r in results
        if r.expected_critical and r.tier != Tier.TIER_3_LOCAL.value
    ]

    over_censor = [
        r for r in results
        if not r.expected_critical and r.tier == Tier.TIER_3_LOCAL.value
    ]

    llm_invoked = [r for r in results if r.llm_invoked]

    finite_k = [
        r.k_lower for r in results
        if r.k_lower is not None and np.isfinite(r.k_lower)
    ]

    mean_k_lower = float(np.mean(finite_k)) if finite_k else 0.0
    median_k_lower = float(np.median(finite_k)) if finite_k else 0.0

    mean_k_by_tier = {}
    for tier_value, tier_name in [
        (Tier.TIER_1_CLOUD_ORIGINAL.value, "TIER_1"),
        (Tier.TIER_2_CLOUD_MASKED.value, "TIER_2"),
        (Tier.TIER_3_LOCAL.value, "TIER_3"),
    ]:
        vals = [
            r.k_lower for r in results
            if r.tier == tier_value and r.k_lower is not None and np.isfinite(r.k_lower)
        ]
        mean_k_by_tier[tier_name] = float(np.mean(vals)) if vals else 0.0

    expected = np.array([1 if r.expected_critical else 0 for r in results])

    correlations = {
        "joint_risk_score": safe_corr(expected, np.array([r.joint_risk_score for r in results])),
        "k_lower": safe_corr(expected, np.array([
            r.k_lower if r.k_lower is not None and np.isfinite(r.k_lower) else 0.0
            for r in results
        ])),
        "direct_identifier_count": safe_corr(expected, np.array([r.direct_identifier_count for r in results])),
        "quasi_identifier_count": safe_corr(expected, np.array([r.quasi_identifier_count for r in results])),
        "masking_ratio": safe_corr(expected, np.array([r.masking_ratio for r in results])),
        "llm_invoked": safe_corr(expected, np.array([1 if r.llm_invoked else 0 for r in results])),
    }

    total_times = [r.total_time for r in results]
    llm_times = [r.llm_time for r in results]

    return BenchmarkAnalysis(
        total_samples=total,

        tier_1_count=len(tier_1),
        tier_2_count=len(tier_2),
        tier_3_count=len(tier_3),

        tier_1_pct=_safe_div(len(tier_1), total) * 100,
        tier_2_pct=_safe_div(len(tier_2), total) * 100,
        tier_3_pct=_safe_div(len(tier_3), total) * 100,

        false_cloud_release_count=len(false_cloud),
        false_cloud_release_rate=_safe_div(len(false_cloud), total) * 100,

        over_censoring_count=len(over_censor),
        over_censoring_rate=_safe_div(len(over_censor), total) * 100,

        llm_invocation_count=len(llm_invoked),
        llm_invocation_rate=_safe_div(len(llm_invoked), total) * 100,

        mean_k_lower=mean_k_lower,
        median_k_lower=median_k_lower,
        mean_k_lower_by_tier=mean_k_by_tier,

        mean_total_time=float(np.mean(total_times)) if total_times else 0.0,
        mean_llm_time=float(np.mean(llm_times)) if llm_times else 0.0,

        correlations=correlations,
    )


# =============================================================================
# Threshold / Weight Sweep Analysis
# =============================================================================

def _normalized_feature_counts(
    results: List[RevisedPipelineResult],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    max_direct = max((r.direct_identifier_count for r in results), default=0)
    max_quasi = max((r.quasi_identifier_count for r in results), default=0)

    direct_norm = {}
    quasi_norm = {}

    for r in results:
        direct_norm[r.sample_hash] = (
            r.direct_identifier_count / max_direct
            if max_direct > 0
            else 0.0
        )
        quasi_norm[r.sample_hash] = (
            r.quasi_identifier_count / max_quasi
            if max_quasi > 0
            else 0.0
        )

    return direct_norm, quasi_norm


def compute_adjusted_risk(
    result: RevisedPipelineResult,
    direct_norm: Dict[str, float],
    quasi_norm: Dict[str, float],
    w_direct: float,
    w_quasi: float,
) -> float:
    base_risk = result.joint_risk_score if result.joint_risk_score is not None else 0.0

    adjusted = (
        base_risk
        + w_direct * direct_norm.get(result.sample_hash, 0.0)
        + w_quasi * quasi_norm.get(result.sample_hash, 0.0)
    )

    return float(np.clip(adjusted, 0.0, 1.0))


def simulate_sensitive_prediction(
    result: RevisedPipelineResult,
    adjusted_risk: float,
    tau_review: float,
    tau_block: float,
    k_minimum: int,
) -> Tuple[bool, bool, bool]:
    """
    Returns:
        marked_sensitive
        llm_review
        blocked

    Interpretation:
    - risk >= tau_block      => blocked/local sensitive
    - tau_review <= risk     => LLM review/local sensitive
    - k_lower < k_minimum    => blocked/local sensitive
    """
    k_lower = result.k_lower if result.k_lower is not None else 0.0

    fails_k = k_lower < k_minimum
    blocked = adjusted_risk >= tau_block or fails_k
    llm_review = tau_review <= adjusted_risk < tau_block and not blocked

    marked_sensitive = blocked or llm_review

    return marked_sensitive, llm_review, blocked


def compute_precision_recall_for_setting(
    results: List[RevisedPipelineResult],
    sweep_name: str,
    tau_review: float,
    tau_block: float,
    k_minimum: int,
    w_direct: float,
    w_quasi: float,
    direct_norm: Dict[str, float],
    quasi_norm: Dict[str, float],
) -> ThresholdSweepPoint:
    tp = fp = tn = fn = 0
    llm_review_count = 0
    block_count = 0

    for r in results:
        adjusted_risk = compute_adjusted_risk(
            r,
            direct_norm=direct_norm,
            quasi_norm=quasi_norm,
            w_direct=w_direct,
            w_quasi=w_quasi,
        )

        predicted_sensitive, llm_review, blocked = simulate_sensitive_prediction(
            r,
            adjusted_risk=adjusted_risk,
            tau_review=tau_review,
            tau_block=tau_block,
            k_minimum=k_minimum,
        )

        actual_sensitive = bool(r.expected_critical)

        if predicted_sensitive and actual_sensitive:
            tp += 1
        elif predicted_sensitive and not actual_sensitive:
            fp += 1
        elif not predicted_sensitive and not actual_sensitive:
            tn += 1
        elif not predicted_sensitive and actual_sensitive:
            fn += 1

        if llm_review:
            llm_review_count += 1
        if blocked:
            block_count += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    false_positive_rate = _safe_div(fp, fp + tn)
    false_negative_rate = _safe_div(fn, fn + tp)

    total = len(results)

    return ThresholdSweepPoint(
        sweep_name=sweep_name,

        tau_review=tau_review,
        tau_block=tau_block,
        k_minimum=k_minimum,
        w_direct=w_direct,
        w_quasi=w_quasi,

        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,

        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,

        marked_sensitive_rate=_safe_div(tp + fp, total),
        llm_review_rate=_safe_div(llm_review_count, total),
        block_rate=_safe_div(block_count, total),
    )


def run_threshold_weight_sweeps(
    results: List[RevisedPipelineResult],
    baseline_tau_review: float = 0.30,
    baseline_tau_block: float = 0.70,
    baseline_k_minimum: int = 5,
    baseline_w_direct: float = 0.50,
    baseline_w_quasi: float = 0.50,
) -> List[ThresholdSweepPoint]:
    if not results:
        return []

    direct_norm, quasi_norm = _normalized_feature_counts(results)
    sweep_points: List[ThresholdSweepPoint] = []

    for tau_review in _frange(0.01, 0.80, 0.02):
        if tau_review >= baseline_tau_block:
            continue

        sweep_points.append(
            compute_precision_recall_for_setting(
                results,
                "tau_review",
                tau_review,
                baseline_tau_block,
                baseline_k_minimum,
                baseline_w_direct,
                baseline_w_quasi,
                direct_norm,
                quasi_norm,
            )
        )

    for tau_block in _frange(0.20, 0.95, 0.05):
        if baseline_tau_review >= tau_block:
            continue

        sweep_points.append(
            compute_precision_recall_for_setting(
                results,
                "tau_block",
                baseline_tau_review,
                tau_block,
                baseline_k_minimum,
                baseline_w_direct,
                baseline_w_quasi,
                direct_norm,
                quasi_norm,
            )
        )

    for k_minimum in range(2, 21):
        sweep_points.append(
            compute_precision_recall_for_setting(
                results,
                "k_minimum",
                baseline_tau_review,
                baseline_tau_block,
                k_minimum,
                baseline_w_direct,
                baseline_w_quasi,
                direct_norm,
                quasi_norm,
            )
        )

    for w_direct in _frange(0.1, 1.0, 0.1):
        sweep_points.append(
            compute_precision_recall_for_setting(
                results,
                "w_direct",
                baseline_tau_review,
                baseline_tau_block,
                baseline_k_minimum,
                w_direct,
                baseline_w_quasi,
                direct_norm,
                quasi_norm,
            )
        )

    for w_quasi in _frange(0.1, 1.0, 0.1):
        sweep_points.append(
            compute_precision_recall_for_setting(
                results,
                "w_quasi",
                baseline_tau_review,
                baseline_tau_block,
                baseline_k_minimum,
                baseline_w_direct,
                w_quasi,
                direct_norm,
                quasi_norm,
            )
        )

    return sweep_points


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(
    results: List[RevisedPipelineResult],
    output_dir: Path,
    baseline_tau_review: float = 0.30,
    baseline_tau_block: float = 0.70,
    baseline_k_minimum: int = 5,
    baseline_w_direct: float = 0.50,
    baseline_w_quasi: float = 0.50,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No results available for visualization.")
        return

    # -------------------------------------------------------------------------
    # Figure 1: Summary plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Revised Pipeline Benchmark Results", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    tier_counts = {
        "TIER_1\nCloud Original": sum(
            1 for r in results if r.tier == Tier.TIER_1_CLOUD_ORIGINAL.value
        ),
        "TIER_2\nCloud Masked": sum(
            1 for r in results if r.tier == Tier.TIER_2_CLOUD_MASKED.value
        ),
        "TIER_3\nLocal": sum(
            1 for r in results if r.tier == Tier.TIER_3_LOCAL.value
        ),
    }

    ax.pie(
        tier_counts.values(),
        labels=tier_counts.keys(),
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
    )
    ax.set_title("Tier Distribution")

    ax = axes[0, 1]
    total = len(results)
    false_cloud = sum(
        1 for r in results
        if r.expected_critical and r.tier != Tier.TIER_3_LOCAL.value
    )
    over_censor = sum(
        1 for r in results
        if not r.expected_critical and r.tier == Tier.TIER_3_LOCAL.value
    )
    llm_invoked = sum(1 for r in results if r.llm_invoked)

    metric_names = ["False Cloud", "Over-censor", "LLM Invoked"]
    metric_values = [
        _safe_div(false_cloud, total) * 100,
        _safe_div(over_censor, total) * 100,
        _safe_div(llm_invoked, total) * 100,
    ]

    ax.bar(metric_names, metric_values, color=["#e74c3c", "#f39c12", "#3498db"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Safety / Operational Metrics")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    finite_k = [
        r.k_lower for r in results
        if r.k_lower is not None and np.isfinite(r.k_lower)
    ]
    if finite_k:
        ax.hist(finite_k, bins=30, color="#9b59b6", alpha=0.8)
    ax.set_xlabel("k_lower")
    ax.set_ylabel("Count")
    ax.set_title("k_lower Distribution")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    risks = [r.joint_risk_score for r in results]
    labels = [1 if r.expected_critical else 0 for r in results]
    colors = ["#e74c3c" if y else "#2ecc71" for y in labels]

    ax.scatter(
        risks,
        [r.direct_identifier_count + r.quasi_identifier_count for r in results],
        c=colors,
        alpha=0.7,
    )
    ax.set_xlabel("Joint Risk Score")
    ax.set_ylabel("Direct + Quasi Identifier Count")
    ax.set_title("Risk vs Entity Count")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "benchmark_summary.png", dpi=300)
    plt.close(fig)

    create_threshold_precision_recall_plots(
        results=results,
        output_dir=output_dir,
        baseline_tau_review=baseline_tau_review,
        baseline_tau_block=baseline_tau_block,
        baseline_k_minimum=baseline_k_minimum,
        baseline_w_direct=baseline_w_direct,
        baseline_w_quasi=baseline_w_quasi,
    )


def create_threshold_precision_recall_plots(
    results: List[RevisedPipelineResult],
    output_dir: Path,
    baseline_tau_review: float = 0.30,
    baseline_tau_block: float = 0.70,
    baseline_k_minimum: int = 5,
    baseline_w_direct: float = 0.50,
    baseline_w_quasi: float = 0.50,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_points = run_threshold_weight_sweeps(
        results=results,
        baseline_tau_review=baseline_tau_review,
        baseline_tau_block=baseline_tau_block,
        baseline_k_minimum=baseline_k_minimum,
        baseline_w_direct=baseline_w_direct,
        baseline_w_quasi=baseline_w_quasi,
    )

    if not sweep_points:
        print("No sweep points generated.")
        return

    _write_json(
        output_dir / "threshold_weight_precision_recall_sweep.json",
        [asdict(p) for p in sweep_points],
    )

    sweep_names = [
        "tau_review",
        "tau_block",
        "k_minimum",
        "w_direct",
        "w_quasi",
    ]

    label_map = {
        "tau_review": r"$\tau_{\mathrm{review}}$",
        "tau_block": r"$\tau_{\mathrm{block}}$",
        "k_minimum": r"$k_{\mathrm{min}}$",
        "w_direct": r"$w_{\mathrm{direct}}$",
        "w_quasi": r"$w_{\mathrm{quasi}}$",
    }

    value_getters = {
        "tau_review": lambda p: p.tau_review,
        "tau_block": lambda p: p.tau_block,
        "k_minimum": lambda p: p.k_minimum,
        "w_direct": lambda p: p.w_direct,
        "w_quasi": lambda p: p.w_quasi,
    }

    # -------------------------------------------------------------------------
    # Precision-recall curves
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    for sweep_name in sweep_names:
        pts = [p for p in sweep_points if p.sweep_name == sweep_name]
        pts = sorted(pts, key=value_getters[sweep_name])

        recalls = [p.recall for p in pts]
        precisions = [p.precision for p in pts]
        values = [value_getters[sweep_name](p) for p in pts]

        ax.plot(
            recalls,
            precisions,
            marker="o",
            linewidth=2,
            markersize=5,
            label=label_map[sweep_name],
        )

        if len(pts) <= 12:
            annotate_indices = range(len(pts))
        else:
            annotate_indices = np.linspace(0, len(pts) - 1, 5, dtype=int)

        for idx in annotate_indices:
            ax.annotate(
                str(values[idx]),
                (recalls[idx], precisions[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Recall / Sensitivity")
    ax.set_ylabel("Precision / Positive Predictive Value")
    ax.set_title("Precision-Recall Curves for Threshold and Weight Sweeps")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "threshold_weight_precision_recall_curves.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Metric-vs-value plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    for ax_idx, sweep_name in enumerate(sweep_names):
        ax = axes[ax_idx]
        pts = [p for p in sweep_points if p.sweep_name == sweep_name]
        pts = sorted(pts, key=value_getters[sweep_name])

        x = [value_getters[sweep_name](p) for p in pts]

        ax.plot(x, [p.precision for p in pts], marker="o", label="Precision")
        ax.plot(x, [p.recall for p in pts], marker="s", label="Recall")
        ax.plot(x, [p.f1 for p in pts], marker="^", label="F1")
        ax.plot(x, [p.false_positive_rate for p in pts], marker="x", label="False Positive Rate")
        ax.plot(x, [p.marked_sensitive_rate for p in pts], marker="D", label="Marked Sensitive Rate")

        ax.set_xlabel(label_map[sweep_name])
        ax.set_ylabel("Metric")
        ax.set_title(f"Metrics vs {label_map[sweep_name]}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].axis("off")

    fig.suptitle("Precision-Recall Metric Sensitivity Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "threshold_weight_metric_sweeps.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # tau_review x tau_block heatmaps
    # -------------------------------------------------------------------------
    tau_review_values = _frange(0.01, 0.50, 0.02)
    tau_block_values = _frange(0.50, 0.95, 0.05)

    direct_norm, quasi_norm = _normalized_feature_counts(results)

    f1_grid = np.full((len(tau_review_values), len(tau_block_values)), np.nan)
    precision_grid = np.full_like(f1_grid, np.nan)
    recall_grid = np.full_like(f1_grid, np.nan)
    fpr_grid = np.full_like(f1_grid, np.nan)

    for i, tau_review in enumerate(tau_review_values):
        for j, tau_block in enumerate(tau_block_values):
            if tau_review >= tau_block:
                continue

            point = compute_precision_recall_for_setting(
                results=results,
                sweep_name="tau_review_tau_block_grid",
                tau_review=tau_review,
                tau_block=tau_block,
                k_minimum=baseline_k_minimum,
                w_direct=baseline_w_direct,
                w_quasi=baseline_w_quasi,
                direct_norm=direct_norm,
                quasi_norm=quasi_norm,
            )

            f1_grid[i, j] = point.f1
            precision_grid[i, j] = point.precision
            recall_grid[i, j] = point.recall
            fpr_grid[i, j] = point.false_positive_rate

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    heatmaps = [
        ("F1 Score", f1_grid, "viridis"),
        ("Precision", precision_grid, "Blues"),
        ("Recall", recall_grid, "Greens"),
        ("False Positive Rate", fpr_grid, "Reds"),
    ]

    for ax, (title, grid, cmap) in zip(axes.flatten(), heatmaps):
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[
                min(tau_block_values),
                max(tau_block_values),
                min(tau_review_values),
                max(tau_review_values),
            ],
        )
        ax.set_xlabel(r"$\tau_{\mathrm{block}}$")
        ax.set_ylabel(r"$\tau_{\mathrm{review}}$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.suptitle(
        r"$\tau_{\mathrm{review}}$ x $\tau_{\mathrm{block}}$ Precision-Recall Heatmaps",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "tau_review_tau_block_heatmaps.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # w_direct x w_quasi heatmaps
    # -------------------------------------------------------------------------
    w_values = _frange(0.1, 1.0, 0.1)

    f1_weight_grid = np.full((len(w_values), len(w_values)), np.nan)
    precision_weight_grid = np.full_like(f1_weight_grid, np.nan)
    recall_weight_grid = np.full_like(f1_weight_grid, np.nan)
    fpr_weight_grid = np.full_like(f1_weight_grid, np.nan)

    for i, w_direct in enumerate(w_values):
        for j, w_quasi in enumerate(w_values):
            point = compute_precision_recall_for_setting(
                results=results,
                sweep_name="weight_grid",
                tau_review=baseline_tau_review,
                tau_block=baseline_tau_block,
                k_minimum=baseline_k_minimum,
                w_direct=w_direct,
                w_quasi=w_quasi,
                direct_norm=direct_norm,
                quasi_norm=quasi_norm,
            )

            f1_weight_grid[i, j] = point.f1
            precision_weight_grid[i, j] = point.precision
            recall_weight_grid[i, j] = point.recall
            fpr_weight_grid[i, j] = point.false_positive_rate

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    heatmaps = [
        ("F1 Score", f1_weight_grid, "viridis"),
        ("Precision", precision_weight_grid, "Blues"),
        ("Recall", recall_weight_grid, "Greens"),
        ("False Positive Rate", fpr_weight_grid, "Reds"),
    ]

    for ax, (title, grid, cmap) in zip(axes.flatten(), heatmaps):
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[
                min(w_values),
                max(w_values),
                min(w_values),
                max(w_values),
            ],
        )
        ax.set_xlabel(r"$w_{\mathrm{quasi}}$")
        ax.set_ylabel(r"$w_{\mathrm{direct}}$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.suptitle(
        r"$w_{\mathrm{direct}}$ x $w_{\mathrm{quasi}}$ Precision-Recall Heatmaps",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "direct_quasi_weight_heatmaps.png", dpi=300)
    plt.close(fig)

    best_by_f1 = sorted(sweep_points, key=lambda p: p.f1, reverse=True)[:10]

    print("\nTop 10 one-dimensional sweep operating points by F1:")
    for p in best_by_f1:
        print(
            f"  {p.sweep_name}: "
            f"tau_review={p.tau_review}, "
            f"tau_block={p.tau_block}, "
            f"k_min={p.k_minimum}, "
            f"w_direct={p.w_direct}, "
            f"w_quasi={p.w_quasi} | "
            f"precision={p.precision:.3f}, "
            f"recall={p.recall:.3f}, "
            f"f1={p.f1:.3f}, "
            f"fpr={p.false_positive_rate:.3f}, "
            f"marked_sensitive_rate={p.marked_sensitive_rate:.3f}"
        )


# =============================================================================
# Loading saved results
# =============================================================================

def load_results_from_output_dir(output_dir: Path) -> List[RevisedPipelineResult]:
    path = output_dir / "pipeline_results.json"

    if not path.exists():
        raise FileNotFoundError(f"No pipeline_results.json found at {path}")

    raw = _read_json(path)
    results = []

    for item in raw:
        item = dict(item)
        item["tiers_by_k"] = {
            int(k): v for k, v in item.get("tiers_by_k", {}).items()
        }
        item["k_lowers_by_k"] = {
            int(k): v for k, v in item.get("k_lowers_by_k", {}).items()
        }
        results.append(RevisedPipelineResult(**item))

    return results


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
    baseline_tau_review: float = 0.30,
    baseline_tau_block: float = 0.70,
    baseline_w_direct: float = 0.50,
    baseline_w_quasi: float = 0.50,
):
    print("=" * 70)
    print("REVISED PRIVACY ROUTER BENCHMARK")
    print("=" * 70)
    print()

    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    print(f"Loading samples from: {input_file}")
    samples = parse_input_file(input_file, max_samples)
    print(f"Loaded {len(samples)} benchmark samples")

    critical_count = sum(1 for s in samples if s.expected_critical)
    print(f"  Expected critical: {critical_count}")
    print(f"  Expected non-critical: {len(samples) - critical_count}")

    if max_samples:
        print(f"Limited to {max_samples} samples")

    print()

    checkpoint = CheckpointManager(out_path, input_file, k_minimum)

    if fresh_start:
        print("Fresh start requested - clearing checkpoint...")
        checkpoint.clear_checkpoint()
    else:
        checkpoint.load_checkpoint()

    pipeline = RevisedBenchmarkPipeline(
        checkpoint_manager=checkpoint,
        k_minimum=k_minimum,
        db_path=db_path,
    )

    print("\n" + "-" * 70)
    print("PROCESSING SAMPLES")
    print("-" * 70)

    results = pipeline.run_all(samples)

    _write_json(out_path / "pipeline_results.json", [asdict(r) for r in results])
    print(f"\nSaved pipeline results to {out_path / 'pipeline_results.json'}")

    if results and not checkpoint.was_interrupted:
        analysis = analyze_results(results)
        _write_json(out_path / "analysis.json", asdict(analysis))

        print("\n" + "-" * 70)
        print("ANALYSIS")
        print("-" * 70)

        print(f"Total samples: {analysis.total_samples}")

        print("\nTier distribution:")
        print(f"  Tier 1: {analysis.tier_1_count} ({analysis.tier_1_pct:.1f}%)")
        print(f"  Tier 2: {analysis.tier_2_count} ({analysis.tier_2_pct:.1f}%)")
        print(f"  Tier 3: {analysis.tier_3_count} ({analysis.tier_3_pct:.1f}%)")

        print("\nSafety / operating metrics:")
        print(
            f"  False cloud releases: "
            f"{analysis.false_cloud_release_count} "
            f"({analysis.false_cloud_release_rate:.2f}%)"
        )
        print(
            f"  Over-censoring: "
            f"{analysis.over_censoring_count} "
            f"({analysis.over_censoring_rate:.2f}%)"
        )
        print(
            f"  LLM invocations: "
            f"{analysis.llm_invocation_count} "
            f"({analysis.llm_invocation_rate:.2f}%)"
        )

        print("\nk-anonymity:")
        print(f"  Mean k_lower: {analysis.mean_k_lower:.2f}")
        print(f"  Median k_lower: {analysis.median_k_lower:.2f}")

        print("\nFeature correlations with expected_critical:")
        for feature, corr in sorted(
            analysis.correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        ):
            print(f"  {feature}: {corr:+.3f}")

        print("\nGenerating visualizations...")
        create_visualizations(
            results=results,
            output_dir=out_path,
            baseline_tau_review=baseline_tau_review,
            baseline_tau_block=baseline_tau_block,
            baseline_k_minimum=k_minimum,
            baseline_w_direct=baseline_w_direct,
            baseline_w_quasi=baseline_w_quasi,
        )

        print(f"\nSaved analysis to {out_path / 'analysis.json'}")

    elif checkpoint.was_interrupted:
        print("\nBenchmark interrupted. Run again to resume.")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark revised privacy-preserving router.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py merged.txt
  python benchmark.py merged.txt --fresh --max-samples 50
  python benchmark.py merged.txt -o results/ -k 5
  python benchmark.py merged.txt --plot-only -o results/

Sweep defaults:
  tau_review: 0.01 to 0.50 step 0.02
  tau_block:  0.50 to 0.95 step 0.05
  k_minimum:  2 to 20 step 1
  w_direct:   0.1 to 1.0 step 0.1
  w_quasi:    0.1 to 1.0 step 0.1
        """,
    )

    parser.add_argument(
        "input_file",
        type=str,
        nargs="?",
        help="Path to benchmark input file.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="benchmark_revised_outputs",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "-n",
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )

    parser.add_argument(
        "-k",
        "--k-minimum",
        type=int,
        default=5,
        help="k-anonymity minimum threshold.",
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="frequency_tables.db",
        help="Path to frequency table database.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and start fresh.",
    )

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmark execution and regenerate plots from saved pipeline_results.json.",
    )

    parser.add_argument(
        "--baseline-tau-review",
        type=float,
        default=0.30,
        help="Baseline tau_review for one-dimensional sweeps.",
    )

    parser.add_argument(
        "--baseline-tau-block",
        type=float,
        default=0.70,
        help="Baseline tau_block for one-dimensional sweeps.",
    )

    parser.add_argument(
        "--baseline-w-direct",
        type=float,
        default=0.50,
        help="Baseline direct identifier weight.",
    )

    parser.add_argument(
        "--baseline-w-quasi",
        type=float,
        default=0.50,
        help="Baseline quasi identifier weight.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.plot_only:
        results = load_results_from_output_dir(output_dir)
        print(f"Loaded {len(results)} saved results from {output_dir / 'pipeline_results.json'}")

        create_visualizations(
            results=results,
            output_dir=output_dir,
            baseline_tau_review=args.baseline_tau_review,
            baseline_tau_block=args.baseline_tau_block,
            baseline_k_minimum=args.k_minimum,
            baseline_w_direct=args.baseline_w_direct,
            baseline_w_quasi=args.baseline_w_quasi,
        )
        return

    if not args.input_file:
        parser.error("input_file is required unless --plot-only is used.")

    run_benchmark(
        input_file=args.input_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        k_minimum=args.k_minimum,
        fresh_start=args.fresh,
        db_path=args.db_path,
        baseline_tau_review=args.baseline_tau_review,
        baseline_tau_block=args.baseline_tau_block,
        baseline_w_direct=args.baseline_w_direct,
        baseline_w_quasi=args.baseline_w_quasi,
    )


if __name__ == "__main__":
    main()