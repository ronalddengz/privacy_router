#!/usr/bin/env python3
"""
privacy_benchmark.py

Combined benchmark + threshold/weight sweep analysis for the privacy-preserving
text router pipeline. Merges the sample-processing/analysis logic of
benchmark.py with the parameter-sweep logic of analysis.py.

Pipeline (same inputs / same operations as benchmark.py):
  1. Parse a benchmark input file into (name, text, expected_critical) samples.
  2. Run each sample through PrivacyRouter, checkpointing progress.
  3. Compute tier distribution / false-cloud-release / over-censoring / LLM
     invocation / k_lower statistics and correlations with expected_critical.
  4. Save UNCERTAIN_NEEDS_LLM records to uncertain_needs_llm.json.

Sweep:
  5. Sweep tau_review, tau_block, k_min, k_lower_threshold, w_direct, and
     w_quasi (one at a time, others held at baseline) and score each
     configuration's precision, recall, F1, false positive rate, and false
     negative rate.
  6. Plot the effect of each parameter on precision/recall and FPR/FNR.

Usage:
    python privacy_benchmark.py example_inputs.txt
    python privacy_benchmark.py example_inputs.txt --fresh --max-samples 50
    python privacy_benchmark.py example_inputs.txt -o results/ -k 5
    python privacy_benchmark.py --sweep-only -o results/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policy import PolicyProfile, Tier, EntityCategory
from frequency_tables import LocalFrequencyTable
from router import PrivacyRouter, RoutingResult


# =============================================================================
# JSON / math helpers
# =============================================================================

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


def _frange(start: float, stop: float, step: float) -> List[float]:
    values = []
    x = start
    while x <= stop + step / 10.0:
        values.append(round(x, 10))
        x += step
    return values


# =============================================================================
# Data classes
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
class PipelineResult:
    """Result from running the pipeline on a single sample."""

    sample_name: str
    sample_hash: str
    text_length: int
    expected_critical: bool

    tier: int
    gate_decision: str
    gate_reasons: List[str]

    k_lower: float
    k_upper: float
    joint_risk_score: float

    direct_identifier_count: int
    quasi_identifier_count: int

    llm_invoked: bool

    original_text_length: int
    masked_text_length: int
    masking_ratio: float

    hard_stop_triggered: bool
    hard_stop_reasons: List[str]
    uncertainty_flags: List[str]

    total_time: float
    detection_time: float
    risk_estimation_time: float
    gate_time: float
    llm_time: float

    processed_at: str = ""

    # Fields for UNCERTAIN_NEEDS_LLM inspection
    pii_masked_text: str = ""
    qi_masked_text: str = ""
    sent_to_local_llm: bool = False
    llm_judgment: Optional[str] = None


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
class OperatingPoint:
    """Metrics for a single swept configuration."""
    sweep_name: str = ""

    tau_review: float = 0.30
    tau_block: float = 0.70
    k_min: int = 5
    k_lower_threshold: int = 20
    w_direct: float = 0.50
    w_quasi: float = 0.50

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    marked_sensitive_rate: float = 0.0
    review_rate: float = 0.0
    block_rate: float = 0.0


# =============================================================================
# Input parsing
# =============================================================================

def parse_input_file(input_file: str, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
    """
    Parse benchmark input file.

    Expected format:

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

        samples.append(BenchmarkSample(name=name, text=sample_text, expected_critical=expected_critical))

        if max_samples is not None and len(samples) >= max_samples:
            break

    if samples:
        return samples

    # Fallback parser.
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    for i, chunk in enumerate(chunks):
        samples.append(BenchmarkSample(name=f"sample_{i + 1}", text=chunk, expected_critical=False))
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


# =============================================================================
# Checkpoint manager
# =============================================================================

class CheckpointManager:
    """Manages saving and loading benchmark progress."""

    def __init__(self, output_dir: Path, input_file: str, k_minimum: int):
        self.output_dir = output_dir
        self.input_file = input_file
        self.k_minimum = k_minimum

        self.checkpoint_file = output_dir / "checkpoint.json"
        self.backup_file = output_dir / "checkpoint.backup.json"

        self.input_hash = self._hash_input_file()

        self.results: Dict[str, PipelineResult] = {}
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

            for sample_hash, raw in data.get("results", {}).items():
                self.results[sample_hash] = PipelineResult(**raw)
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
            "results": {h: asdict(r) for h, r in self.results.items()},
        }

        if self.checkpoint_file.exists():
            self.checkpoint_file.replace(self.backup_file)

        _write_json(self.checkpoint_file, data)

    def add_result(self, sample_hash: str, result: PipelineResult):
        self.results[sample_hash] = result
        self.processed_hashes.add(sample_hash)

    def is_processed(self, sample_hash: str) -> bool:
        return sample_hash in self.processed_hashes

    def get_result(self, sample_hash: str) -> Optional[PipelineResult]:
        return self.results.get(sample_hash)


# =============================================================================
# Pipeline runner
# =============================================================================

class BenchmarkPipeline:
    """Runs the privacy router pipeline with checkpointing."""

    def __init__(self, checkpoint_manager: CheckpointManager, k_minimum: int = 5, db_path: str = "frequency_tables.db"):
        self.checkpoint = checkpoint_manager
        self.k_minimum = k_minimum
        self.db_path = db_path

        print("Initializing frequency table...")
        db_path_obj = Path(self.db_path)
        if not db_path_obj.exists():
            print(f"  Creating sample frequency table at {db_path_obj}...")
        else:
            print(f"  Refreshing sample frequency table at {db_path_obj}...")
        self.freq_table = LocalFrequencyTable(str(db_path_obj))

        print("Initializing privacy router...")
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
        self.router = PrivacyRouter(policy=policy, frequency_table=self.freq_table)

    def process_sample(self, sample: BenchmarkSample) -> PipelineResult:
        start = time.time()

        result: RoutingResult = self.router.route(sample.text)

        total_time = time.time() - start
        llm_time = getattr(result, "llm_time", None) if getattr(result, "llm_time", None) is not None else 0.0
        non_llm_time = max(total_time - llm_time, 0.0)

        detection_time = non_llm_time * 0.40
        risk_estimation_time = non_llm_time * 0.30
        gate_time = non_llm_time * 0.30

        pii_count = len(getattr(result, "pii_evidence", []) or [])
        qi_count = len(getattr(result, "qi_evidence", []) or [])

        direct_identifier_count = len([
            e for e in (getattr(result, "pii_evidence", []) or [])
            if getattr(e, "category", None) == EntityCategory.DIRECT_IDENTIFIER
        ])
        if direct_identifier_count == 0:
            direct_identifier_count = pii_count

        is_uncertain_llm = getattr(result.decision, "contextual_review_invoked", False)

        if result.decision.tier == Tier.TIER_3_LOCAL:
            gate_decision = "UNCERTAIN_NEEDS_LLM" if is_uncertain_llm else "UNSAFE_ROUTE_LOCAL"
        else:
            gate_decision = "SAFE_FOR_CLOUD"

        gate_reasons = getattr(result.decision, "hard_stop_reasons", []) or []

        k_lower = (
            result.risk_estimate.lower_bound_k
            if getattr(result, "risk_estimate", None) and getattr(result.risk_estimate, "lower_bound_k", None) is not None
            else 0.0
        )
        k_upper = (
            result.risk_estimate.upper_bound_k
            if getattr(result, "risk_estimate", None) and getattr(result.risk_estimate, "upper_bound_k", None) is not None
            else float("inf")
        )
        joint_risk = 0.0 if not k_lower or k_lower == float("inf") else min(1.0, 1.0 / k_lower)

        original_len = len(sample.text)
        masked_text = getattr(result, "masked_text", sample.text) or sample.text
        masked_len = len(masked_text)
        masking_ratio = (original_len - masked_len) / original_len if original_len > 0 else 0.0

        # Extract PII-masked text safely
        pii_masked_text = getattr(result, "pii_masked_text", None)
        if not pii_masked_text:
            if hasattr(self.router, "mask_pii"):
                pii_masked_text = self.router.mask_pii(sample.text)
            else:
                pii_masked_text = getattr(result, "masked_text", sample.text) or sample.text

        # Extract QI-masked text safely
        qi_masked_text = getattr(result, "qi_masked_text", None)
        if not qi_masked_text:
            if hasattr(self.router, "mask_qi"):
                qi_masked_text = self.router.mask_qi(sample.text)
            else:
                qi_masked_text = getattr(result, "masked_text", sample.text) or sample.text

        sent_to_local_llm = getattr(result, "sent_to_local_llm", is_uncertain_llm)
        llm_judgment = getattr(
            result,
            "llm_judgment",
            getattr(result.decision, "contextual_review_decision", getattr(result, "llm_response", None))
        )

        return PipelineResult(
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
            llm_invoked=is_uncertain_llm,
            original_text_length=original_len,
            masked_text_length=masked_len,
            masking_ratio=masking_ratio,
            hard_stop_triggered=bool(getattr(result.decision, "hard_stop_reasons", [])),
            hard_stop_reasons=getattr(result.decision, "hard_stop_reasons", []) or [],
            uncertainty_flags=getattr(result.decision, "uncertainty_flags", []) or [],
            total_time=total_time,
            detection_time=detection_time,
            risk_estimation_time=risk_estimation_time,
            gate_time=gate_time,
            llm_time=llm_time,
            processed_at=datetime.now().isoformat(),
            pii_masked_text=pii_masked_text,
            qi_masked_text=qi_masked_text,
            sent_to_local_llm=sent_to_local_llm,
            llm_judgment=llm_judgment,
        )

    def run_all(self, samples: List[BenchmarkSample], save_interval: int = 1) -> List[PipelineResult]:
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


def load_results_from_output_dir(output_dir: Path) -> List[PipelineResult]:
    path = output_dir / "pipeline_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No pipeline_results.json found at {path}")
    raw = _read_json(path)
    return [PipelineResult(**item) for item in raw]


# =============================================================================
# Analysis
# =============================================================================

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def analyze_results(results: List[PipelineResult]) -> BenchmarkAnalysis:
    total = len(results)

    tier_1 = [r for r in results if r.tier == Tier.TIER_1_CLOUD_ORIGINAL.value]
    tier_2 = [r for r in results if r.tier == Tier.TIER_2_CLOUD_MASKED.value]
    tier_3 = [r for r in results if r.tier == Tier.TIER_3_LOCAL.value]

    false_cloud = [r for r in results if r.expected_critical and r.tier != Tier.TIER_3_LOCAL.value]
    over_censor = [r for r in results if not r.expected_critical and r.tier == Tier.TIER_3_LOCAL.value]
    llm_invoked = [r for r in results if r.llm_invoked]

    finite_k = [r.k_lower for r in results if r.k_lower is not None and np.isfinite(r.k_lower)]
    mean_k_lower = float(np.mean(finite_k)) if finite_k else 0.0
    median_k_lower = float(np.median(finite_k)) if finite_k else 0.0

    mean_k_by_tier = {}
    for tier_value, tier_name in [
        (Tier.TIER_1_CLOUD_ORIGINAL.value, "TIER_1"),
        (Tier.TIER_2_CLOUD_MASKED.value, "TIER_2"),
        (Tier.TIER_3_LOCAL.value, "TIER_3"),
    ]:
        vals = [r.k_lower for r in results if r.tier == tier_value and r.k_lower is not None and np.isfinite(r.k_lower)]
        mean_k_by_tier[tier_name] = float(np.mean(vals)) if vals else 0.0

    expected = np.array([1 if r.expected_critical else 0 for r in results])
    correlations = {
        "joint_risk_score": safe_corr(expected, np.array([r.joint_risk_score for r in results])),
        "k_lower": safe_corr(expected, np.array([
            r.k_lower if r.k_lower is not None and np.isfinite(r.k_lower) else 0.0 for r in results
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


def plot_benchmark_summary(results: List[PipelineResult], output_dir: Path) -> None:
    """Summary figure: tier split, safety metrics, k_lower distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pipeline Benchmark Results", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    tier_counts = {
        "TIER_1\nCloud Original": sum(1 for r in results if r.tier == Tier.TIER_1_CLOUD_ORIGINAL.value),
        "TIER_2\nCloud Masked": sum(1 for r in results if r.tier == Tier.TIER_2_CLOUD_MASKED.value),
        "TIER_3\nLocal": sum(1 for r in results if r.tier == Tier.TIER_3_LOCAL.value),
    }
    ax.pie(tier_counts.values(), labels=tier_counts.keys(), autopct="%1.1f%%", startangle=90,
           colors=["#2ecc71", "#f39c12", "#e74c3c"])
    ax.set_title("Tier Distribution")

    ax = axes[0, 1]
    total = len(results)
    false_cloud = sum(1 for r in results if r.expected_critical and r.tier != Tier.TIER_3_LOCAL.value)
    over_censor = sum(1 for r in results if not r.expected_critical and r.tier == Tier.TIER_3_LOCAL.value)
    llm_invoked = sum(1 for r in results if r.llm_invoked)
    ax.bar(
        ["False Cloud", "Over-censor", "LLM Invoked"],
        [_safe_div(false_cloud, total) * 100, _safe_div(over_censor, total) * 100, _safe_div(llm_invoked, total) * 100],
        color=["#e74c3c", "#f39c12", "#3498db"],
    )
    ax.set_ylabel("Rate (%)")
    ax.set_title("Safety / Operational Metrics")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    finite_k = [r.k_lower for r in results if r.k_lower is not None and np.isfinite(r.k_lower)]
    if finite_k:
        ax.hist(finite_k, bins=30, color="#9b59b6", alpha=0.8)
    ax.set_xlabel("k_lower")
    ax.set_ylabel("Count")
    ax.set_title("k_lower Distribution")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    risks = [r.joint_risk_score for r in results]
    colors = ["#e74c3c" if r.expected_critical else "#2ecc71" for r in results]
    ax.scatter(risks, [r.direct_identifier_count + r.quasi_identifier_count for r in results], c=colors, alpha=0.7)
    ax.set_xlabel("Joint Risk Score")
    ax.set_ylabel("Direct + Quasi Identifier Count")
    ax.set_title("Risk vs Entity Count")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "benchmark_summary.png", dpi=300)
    plt.close(fig)
    print("  Saved benchmark_summary.png")


# =============================================================================
# Parameter sweep (tau_review, tau_block, k_min, k_lower_threshold, w_direct, w_quasi)
# =============================================================================

def compute_normalization(results: List[PipelineResult]) -> Tuple[float, float]:
    max_direct = max((r.direct_identifier_count for r in results), default=1)
    max_quasi = max((r.quasi_identifier_count for r in results), default=1)
    return float(max(max_direct, 1)), float(max(max_quasi, 1))


def simulate_routing(
    result: PipelineResult,
    max_direct: float,
    max_quasi: float,
    tau_review: float,
    tau_block: float,
    k_min: int,
    k_lower_threshold: float,
    w_direct: float,
    w_quasi: float,
) -> Tuple[bool, bool, bool]:
    """
    Simulate a routing decision for one configuration.

    Returns: (marked_sensitive, is_review, is_block)
    """
    direct_score = result.direct_identifier_count / max_direct
    quasi_score = result.quasi_identifier_count / max_quasi
    combined_risk = float(np.clip(result.joint_risk_score + w_direct * direct_score + w_quasi * quasi_score, 0.0, 1.0))

    k_lower = result.k_lower if result.k_lower is not None else 0.0

    # Hard block: k-anonymity below the minimum.
    if k_lower < k_min:
        return True, False, True

    # Block: combined risk above the block threshold.
    if combined_risk >= tau_block:
        return True, False, True

    # Review: combined risk above the review threshold.
    if combined_risk >= tau_review:
        return True, True, False

    # Review: k-anonymity below the safe threshold (but still >= k_min).
    if k_lower < k_lower_threshold:
        return True, True, False

    return False, False, False


def evaluate_operating_point(
    results: List[PipelineResult],
    max_direct: float,
    max_quasi: float,
    sweep_name: str,
    tau_review: float,
    tau_block: float,
    k_min: int,
    k_lower_threshold: float,
    w_direct: float,
    w_quasi: float,
) -> OperatingPoint:
    tp = fp = tn = fn = 0
    review_count = 0
    block_count = 0

    for r in results:
        marked, is_review, is_block = simulate_routing(
            r, max_direct, max_quasi, tau_review, tau_block, k_min, k_lower_threshold, w_direct, w_quasi
        )
        actual = r.expected_critical

        if marked and actual:
            tp += 1
        elif marked and not actual:
            fp += 1
        elif not marked and not actual:
            tn += 1
        else:
            fn += 1

        if is_review:
            review_count += 1
        if is_block:
            block_count += 1

    total = len(results)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)

    return OperatingPoint(
        sweep_name=sweep_name,
        tau_review=tau_review,
        tau_block=tau_block,
        k_min=k_min,
        k_lower_threshold=k_lower_threshold,
        w_direct=w_direct,
        w_quasi=w_quasi,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        marked_sensitive_rate=_safe_div(tp + fp, total),
        review_rate=_safe_div(review_count, total),
        block_rate=_safe_div(block_count, total),
    )


TAU_REVIEW_VALUES = _frange(0.02, 0.95, 0.05)
TAU_BLOCK_VALUES = _frange(0.02, 0.95, 0.05)
K_MIN_VALUES = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
K_LOWER_THRESHOLD_VALUES = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
WEIGHT_VALUES = _frange(0.1, 1.0, 0.1)


def run_param_sweeps(
    results: List[PipelineResult],
    baseline_tau_review: float,
    baseline_tau_block: float,
    baseline_k_min: int,
    baseline_k_lower_threshold: float,
    baseline_w_direct: float,
    baseline_w_quasi: float,
) -> Dict[str, List[OperatingPoint]]:
    """Sweep each parameter one at a time, holding others at baseline."""
    max_direct, max_quasi = compute_normalization(results)

    def eval_point(name: str, **overrides) -> OperatingPoint:
        params = dict(
            tau_review=baseline_tau_review,
            tau_block=baseline_tau_block,
            k_min=baseline_k_min,
            k_lower_threshold=baseline_k_lower_threshold,
            w_direct=baseline_w_direct,
            w_quasi=baseline_w_quasi,
        )
        params.update(overrides)
        return evaluate_operating_point(results, max_direct, max_quasi, name, **params)

    sweeps: Dict[str, List[OperatingPoint]] = {}

    sweeps["tau_review"] = [
        eval_point("tau_review", tau_review=v) for v in TAU_REVIEW_VALUES if v < baseline_tau_block
    ]
    sweeps["tau_block"] = [
        eval_point("tau_block", tau_block=v, w_direct=0.2, w_quasi=0.2) for v in TAU_BLOCK_VALUES if v > baseline_tau_review
    ]
    sweeps["k_min"] = [
        eval_point("k_min_strict", k_min=v, k_lower_threshold=v) 
        for v in K_MIN_VALUES
    ]
    sweeps["k_lower_threshold"] = [
        eval_point("k_lower_threshold", k_lower_threshold=v)
        for v in K_LOWER_THRESHOLD_VALUES
        if v >= baseline_k_min
    ]
    sweeps["w_direct"] = [eval_point("w_direct", w_direct=v) for v in WEIGHT_VALUES]
    sweeps["w_quasi"] = [eval_point("w_quasi", w_quasi=v) for v in WEIGHT_VALUES]

    return sweeps


# =============================================================================
# Sweep plotting
# =============================================================================

PARAM_LABELS = {
    "tau_review": r"$\tau_{\mathrm{review}}$",
    "tau_block": r"$\tau_{\mathrm{block}}$",
    "k_min": r"$k_{\mathrm{min}}$",
    "k_lower_threshold": r"$k_{\mathrm{lower\_threshold}}$",
    "w_direct": r"$w_{\mathrm{direct}}$",
    "w_quasi": r"$w_{\mathrm{quasi}}$",
}

PARAM_ATTR = {
    "tau_review": "tau_review",
    "tau_block": "tau_block",
    "k_min": "k_min",
    "k_lower_threshold": "k_lower_threshold",
    "w_direct": "w_direct",
    "w_quasi": "w_quasi",
}


def plot_param_sweep(points: List[OperatingPoint], param_name: str, output_dir: Path) -> None:
    if not points:
        return

    attr = PARAM_ATTR[param_name]
    label = PARAM_LABELS[param_name]
    points = sorted(points, key=lambda p: getattr(p, attr))
    x = [getattr(p, attr) for p in points]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    ax.plot(x, [p.precision for p in points], "b-o", label="Precision", linewidth=2)
    ax.plot(x, [p.recall for p in points], "g-s", label="Recall", linewidth=2)
    ax.plot(x, [p.f1 for p in points], "k--^", label="F1", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall vs {label}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, [p.false_positive_rate for p in points], "m-o", label="False Positive Rate", linewidth=2)
    ax.plot(x, [p.false_negative_rate for p in points], "r-s", label="False Negative Rate", linewidth=2)
    ax.set_xlabel(label)
    ax.set_ylabel("Rate")
    ax.set_title(f"FPR / FNR vs {label}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filename = f"sweep_{param_name}.png"
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def create_sweep_plots(sweeps: Dict[str, List[OperatingPoint]], output_dir: Path) -> None:
    print("Creating sweep plots...")
    for param_name, points in sweeps.items():
        plot_param_sweep(points, param_name, output_dir)


# =============================================================================
# Ablation Study: Effect of PII and QI Detectors
# =============================================================================

@dataclass
class AblationResult:
    """Result from an ablation experiment."""
    condition: str
    description: str

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    tier_1_rate: float = 0.0
    tier_2_rate: float = 0.0
    tier_3_rate: float = 0.0

    mean_combined_risk: float = 0.0


def simulate_routing_ablation(
    result: PipelineResult,
    max_direct: float,
    max_quasi: float,
    tau_review: float,
    tau_block: float,
    k_min: int,
    k_lower_threshold: float,
    w_direct: float,
    w_quasi: float,
    use_pii: bool = True,
    use_qi: bool = True,
) -> Tuple[bool, bool, bool, float]:
    """Simulate a routing decision with optional ablation of PII/QI detectors."""
    direct_count = result.direct_identifier_count if use_pii else 0
    quasi_count = result.quasi_identifier_count if use_qi else 0

    direct_score = direct_count / max_direct if max_direct > 0 else 0.0
    quasi_score = quasi_count / max_quasi if max_quasi > 0 else 0.0

    base_risk = result.joint_risk_score if (use_pii or use_qi) else 0.0

    combined_risk = float(np.clip(
        base_risk + w_direct * direct_score + w_quasi * quasi_score,
        0.0,
        1.0
    ))

    k_lower = result.k_lower if result.k_lower is not None else 0.0

    if use_pii or use_qi:
        if k_lower < k_min:
            return True, False, True, combined_risk

    if combined_risk >= tau_block:
        return True, False, True, combined_risk

    if combined_risk >= tau_review:
        return True, True, False, combined_risk

    if (use_pii or use_qi) and k_lower < k_lower_threshold:
        return True, True, False, combined_risk

    return False, False, False, combined_risk


def evaluate_ablation_condition(
    results: List[PipelineResult],
    max_direct: float,
    max_quasi: float,
    tau_review: float,
    tau_block: float,
    k_min: int,
    k_lower_threshold: float,
    w_direct: float,
    w_quasi: float,
    condition: str,
    use_pii: bool,
    use_qi: bool,
) -> AblationResult:
    tp = fp = tn = fn = 0
    tier_1 = tier_2 = tier_3 = 0
    combined_risks = []

    for r in results:
        marked, is_review, is_block, combined_risk = simulate_routing_ablation(
            r, max_direct, max_quasi, tau_review, tau_block,
            k_min, k_lower_threshold, w_direct, w_quasi,
            use_pii=use_pii, use_qi=use_qi
        )
        combined_risks.append(combined_risk)
        actual = r.expected_critical

        if marked and actual:
            tp += 1
        elif marked and not actual:
            fp += 1
        elif not marked and not actual:
            tn += 1
        else:
            fn += 1

        if is_block:
            tier_3 += 1
        elif is_review:
            tier_2 += 1
        else:
            tier_1 += 1

    total = len(results)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)

    descriptions = {
        "full": "Full system (PII + QI detectors)",
        "no_pii": "QI detector only (PII disabled)",
        "no_qi": "PII detector only (QI disabled)",
        "no_detectors": "No detectors (baseline)",
    }

    return AblationResult(
        condition=condition,
        description=descriptions.get(condition, condition),
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        tier_1_rate=_safe_div(tier_1, total),
        tier_2_rate=_safe_div(tier_2, total),
        tier_3_rate=_safe_div(tier_3, total),
        mean_combined_risk=float(np.mean(combined_risks)) if combined_risks else 0.0,
    )


def run_ablation_study(
    results: List[PipelineResult],
    tau_review: float,
    tau_block: float,
    k_min: int,
    k_lower_threshold: float,
    w_direct: float,
    w_quasi: float,
) -> Dict[str, AblationResult]:
    max_direct, max_quasi = compute_normalization(results)

    ablation_configs = [
        ("full", True, True),
        ("no_pii", False, True),
        ("no_qi", True, False),
        ("no_detectors", False, False),
    ]

    ablation_results = {}
    for condition, use_pii, use_qi in ablation_configs:
        ablation_results[condition] = evaluate_ablation_condition(
            results, max_direct, max_quasi,
            tau_review, tau_block, k_min, k_lower_threshold,
            w_direct, w_quasi,
            condition, use_pii, use_qi
        )

    return ablation_results


def plot_ablation_study(ablation_results: Dict[str, AblationResult], output_dir: Path) -> None:
    conditions = ["full", "no_pii", "no_qi", "no_detectors"]
    labels = [
        "Full System\n(PII + QI)",
        "QI Only\n(No PII)",
        "PII Only\n(No QI)",
        "No Detectors\n(Baseline)"
    ]

    precisions = [ablation_results[c].precision for c in conditions]
    recalls = [ablation_results[c].recall for c in conditions]
    f1s = [ablation_results[c].f1 for c in conditions]
    fprs = [ablation_results[c].false_positive_rate for c in conditions]
    fnrs = [ablation_results[c].false_negative_rate for c in conditions]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Ablation Study: Effect of PII and QI Detectors", fontsize=14, fontweight="bold")

    x = np.arange(len(conditions))
    width = 0.25

    ax = axes[0, 0]
    bars1 = ax.bar(x - width, precisions, width, label="Precision", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label="Recall", color="#2ecc71", alpha=0.8)
    bars3 = ax.bar(x + width, f1s, width, label="F1", color="#9b59b6", alpha=0.8)
    ax.set_ylabel("Score")
    ax.set_title("Classification Performance by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, fprs, width, label="False Positive Rate", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width/2, fnrs, width, label="False Negative Rate", color="#f39c12", alpha=0.8)
    ax.set_ylabel("Rate")
    ax.set_title("Error Rates by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax = axes[1, 0]
    tier_1_rates = [ablation_results[c].tier_1_rate * 100 for c in conditions]
    tier_2_rates = [ablation_results[c].tier_2_rate * 100 for c in conditions]
    tier_3_rates = [ablation_results[c].tier_3_rate * 100 for c in conditions]

    ax.bar(x, tier_1_rates, width=0.6, label="Tier 1 (Cloud Original)", color="#2ecc71", alpha=0.8)
    ax.bar(x, tier_2_rates, width=0.6, bottom=tier_1_rates, label="Tier 2 (Cloud Masked)", color="#f39c12", alpha=0.8)
    ax.bar(x, tier_3_rates, width=0.6, bottom=[t1 + t2 for t1, t2 in zip(tier_1_rates, tier_2_rates)],
           label="Tier 3 (Local)", color="#e74c3c", alpha=0.8)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Tier Distribution by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    full_f1 = ablation_results["full"].f1
    deltas = [(ablation_results[c].f1 - full_f1) * 100 for c in conditions]
    colors = ["#3498db" if d >= 0 else "#e74c3c" for d in deltas]
    bars = ax.bar(x, deltas, width=0.6, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("F1 Change (%)")
    ax.set_title("F1 Score Change vs Full System")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -3
        ax.annotate(f'{delta:+.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, offset), textcoords="offset points", ha='center', va=va, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ablation_study.png")


def plot_ablation_contribution(ablation_results: Dict[str, AblationResult], output_dir: Path) -> None:
    full = ablation_results["full"]
    no_pii = ablation_results["no_pii"]
    no_qi = ablation_results["no_qi"]
    no_detectors = ablation_results["no_detectors"]

    metrics = ["Precision", "Recall", "F1", "1-FPR", "1-FNR"]

    pii_contrib = [
        full.precision - no_pii.precision,
        full.recall - no_pii.recall,
        full.f1 - no_pii.f1,
        (1 - full.false_positive_rate) - (1 - no_pii.false_positive_rate),
        (1 - full.false_negative_rate) - (1 - no_pii.false_negative_rate),
    ]

    qi_contrib = [
        full.precision - no_qi.precision,
        full.recall - no_qi.recall,
        full.f1 - no_qi.f1,
        (1 - full.false_positive_rate) - (1 - no_qi.false_positive_rate),
        (1 - full.false_negative_rate) - (1 - no_qi.false_negative_rate),
    ]

    interaction = [
        full.precision - no_pii.precision - no_qi.precision + no_detectors.precision,
        full.recall - no_pii.recall - no_qi.recall + no_detectors.recall,
        full.f1 - no_pii.f1 - no_qi.f1 + no_detectors.f1,
        (1 - full.false_positive_rate) - (1 - no_pii.false_positive_rate) - (1 - no_qi.false_positive_rate) + (1 - no_detectors.false_positive_rate),
        (1 - full.false_negative_rate) - (1 - no_pii.false_negative_rate) - (1 - no_qi.false_negative_rate) + (1 - no_detectors.false_negative_rate),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Detector Contribution Analysis", fontsize=14, fontweight="bold")

    x = np.arange(len(metrics))
    width = 0.25

    ax = axes[0]
    ax.bar(x - width, pii_contrib, width, label="PII Detector", color="#e74c3c", alpha=0.8)
    ax.bar(x, qi_contrib, width, label="QI Detector", color="#3498db", alpha=0.8)
    ax.bar(x + width, interaction, width, label="Interaction", color="#9b59b6", alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Contribution to Score")
    ax.set_title("Individual Detector Contributions")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    full_metrics = [full.precision, full.recall, full.f1, 1 - full.false_positive_rate, 1 - full.false_negative_rate]

    pii_pct = [_safe_div(c, f) * 100 for c, f in zip(pii_contrib, full_metrics)]
    qi_pct = [_safe_div(c, f) * 100 for c, f in zip(qi_contrib, full_metrics)]

    ax.bar(x - width/2, pii_pct, width, label="PII Detector", color="#e74c3c", alpha=0.8)
    ax.bar(x + width/2, qi_pct, width, label="QI Detector", color="#3498db", alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Relative Contribution (%)")
    ax.set_title("Relative Importance of Each Detector")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "ablation_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ablation_contribution.png")


def create_ablation_plots(ablation_results: Dict[str, AblationResult], output_dir: Path) -> None:
    print("Creating ablation study plots...")
    plot_ablation_study(ablation_results, output_dir)
    plot_ablation_contribution(ablation_results, output_dir)


# =============================================================================
# Main pipeline + sweep runner
# =============================================================================

def run_benchmark_analysis(
    input_file: Optional[str],
    output_dir: str = "benchmark_outputs",
    max_samples: Optional[int] = None,
    k_minimum: int = 5,
    fresh_start: bool = False,
    db_path: str = "frequency_tables.db",
    sweep_only: bool = False,
    baseline_tau_review: float = 0.30,
    baseline_tau_block: float = 0.70,
    baseline_k_lower_threshold: float = 20,
    baseline_w_direct: float = 0.50,
    baseline_w_quasi: float = 0.50,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    if sweep_only:
        print(f"Loading saved results from {out_path / 'pipeline_results.json'}...")
        results = load_results_from_output_dir(out_path)
        print(f"  Loaded {len(results)} results")
    else:
        print("=" * 70)
        print("PRIVACY ROUTER BENCHMARK")
        print("=" * 70)

        print(f"\nLoading samples from: {input_file}")
        samples = parse_input_file(input_file, max_samples)
        print(f"Loaded {len(samples)} benchmark samples")

        critical_count = sum(1 for s in samples if s.expected_critical)
        print(f"  Expected critical: {critical_count}")
        print(f"  Expected non-critical: {len(samples) - critical_count}")
        if max_samples:
            print(f"Limited to {max_samples} samples")

        checkpoint = CheckpointManager(out_path, input_file, k_minimum)
        if fresh_start:
            print("Fresh start requested - clearing checkpoint...")
            checkpoint.clear_checkpoint()
        else:
            checkpoint.load_checkpoint()

        pipeline = BenchmarkPipeline(checkpoint_manager=checkpoint, k_minimum=k_minimum, db_path=db_path)

        print("\n" + "-" * 70)
        print("PROCESSING SAMPLES")
        print("-" * 70)
        results = pipeline.run_all(samples)

        _write_json(out_path / "pipeline_results.json", [asdict(r) for r in results])
        print(f"\nSaved pipeline results to {out_path / 'pipeline_results.json'}")

        if checkpoint.was_interrupted:
            print("\nBenchmark interrupted. Run again to resume.")
            return

    if not results:
        print("No results available for analysis.")
        return

    # Export json for inputs marked as UNCERTAIN_NEEDS_LLM
    uncertain_llm_records = [
        {
            "sample_name": r.sample_name,
            "sample_hash": r.sample_hash,
            "text_after_pii_masking": r.pii_masked_text,
            "text_after_qi_masking": r.qi_masked_text,
            "was_sent_to_local_llm": r.sent_to_local_llm,
            "llm_judgment": r.llm_judgment,
        }
        for r in results
        if r.gate_decision in ("UNCERTAIN_NEEDS_LLM", "UNCERTAIN_NEED_LLM")
    ]
    _write_json(out_path / "uncertain_needs_llm.json", uncertain_llm_records)
    print(f"Saved {len(uncertain_llm_records)} UNCERTAIN_NEEDS_LLM record(s) to {out_path / 'uncertain_needs_llm.json'}")

    # Standard benchmark analysis
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
    print(f"  False cloud releases: {analysis.false_cloud_release_count} ({analysis.false_cloud_release_rate:.2f}%)")
    print(f"  Over-censoring: {analysis.over_censoring_count} ({analysis.over_censoring_rate:.2f}%)")
    print(f"  LLM invocations: {analysis.llm_invocation_count} ({analysis.llm_invocation_rate:.2f}%)")
    print("\nk-anonymity:")
    print(f"  Mean k_lower: {analysis.mean_k_lower:.2f}")
    print(f"  Median k_lower: {analysis.median_k_lower:.2f}")
    print("\nFeature correlations with expected_critical:")
    for feature, corr in sorted(analysis.correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature}: {corr:+.3f}")

    print("\nGenerating benchmark summary plot...")
    plot_benchmark_summary(results, out_path)

    # Ablation Study
    print("\n" + "-" * 70)
    print("ABLATION STUDY")
    print("-" * 70)
    print("Testing effect of PII and QI detectors...")

    ablation_results = run_ablation_study(
        results,
        tau_review=baseline_tau_review,
        tau_block=baseline_tau_block,
        k_min=k_minimum,
        k_lower_threshold=baseline_k_lower_threshold,
        w_direct=baseline_w_direct,
        w_quasi=baseline_w_quasi,
    )

    ablation_json = {name: asdict(r) for name, r in ablation_results.items()}
    _write_json(out_path / "ablation_results.json", ablation_json)
    print(f"Saved ablation results to {out_path / 'ablation_results.json'}")

    create_ablation_plots(ablation_results, out_path)

    print("\nAblation Study Results:")
    print(f"{'Condition':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10} {'FNR':>10}")
    print("-" * 70)
    for condition in ["full", "no_pii", "no_qi", "no_detectors"]:
        r = ablation_results[condition]
        print(f"{r.description:<20} {r.precision:>10.3f} {r.recall:>10.3f} {r.f1:>10.3f} "
              f"{r.false_positive_rate:>10.3f} {r.false_negative_rate:>10.3f}")

    full = ablation_results["full"]
    no_pii = ablation_results["no_pii"]
    no_qi = ablation_results["no_qi"]

    pii_contrib_f1 = full.f1 - no_pii.f1
    qi_contrib_f1 = full.f1 - no_qi.f1

    print("\nDetector Contributions (F1 score change when removed):")
    print(f"  PII Detector: {pii_contrib_f1:+.4f} ({_safe_div(pii_contrib_f1, full.f1) * 100:+.1f}% of full system)")
    print(f"  QI Detector:  {qi_contrib_f1:+.4f} ({_safe_div(qi_contrib_f1, full.f1) * 100:+.1f}% of full system)")

    # Parameter sweep
    print("\n" + "-" * 70)
    print("PARAMETER SWEEP")
    print("-" * 70)
    print("Sweeping tau_review, tau_block, k_min, k_lower_threshold, w_direct, w_quasi...")

    sweeps = run_param_sweeps(
        results,
        baseline_tau_review=baseline_tau_review,
        baseline_tau_block=baseline_tau_block,
        baseline_k_min=k_minimum,
        baseline_k_lower_threshold=baseline_k_lower_threshold,
        baseline_w_direct=baseline_w_direct,
        baseline_w_quasi=baseline_w_quasi,
    )

    sweep_json = {name: [asdict(p) for p in points] for name, points in sweeps.items()}
    _write_json(out_path / "sweep_results.json", sweep_json)
    print(f"Saved sweep results to {out_path / 'sweep_results.json'}")

    create_sweep_plots(sweeps, out_path)

    all_points = [p for pts in sweeps.values() for p in pts]
    best_by_f1 = sorted(all_points, key=lambda p: p.f1, reverse=True)[:10]
    print("\nTop 10 sweep operating points by F1:")
    for p in best_by_f1:
        print(
            f"  [{p.sweep_name}] tau_review={p.tau_review}, tau_block={p.tau_block}, "
            f"k_min={p.k_min}, k_lower_threshold={p.k_lower_threshold}, "
            f"w_direct={p.w_direct}, w_quasi={p.w_quasi} | "
            f"precision={p.precision:.3f}, recall={p.recall:.3f}, f1={p.f1:.3f}, "
            f"fpr={p.false_positive_rate:.3f}, fnr={p.false_negative_rate:.3f}"
        )

    print("\n" + "=" * 70)
    print(f"COMPLETE. Results in {out_path}/")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the privacy router benchmark and sweep tau_review, tau_block, k_min, "
                     "k_lower_threshold, w_direct, and w_quasi.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python privacy_benchmark.py merged.txt
  python privacy_benchmark.py merged.txt --fresh --max-samples 50
  python privacy_benchmark.py merged.txt -o results/ -k 5
  python privacy_benchmark.py --sweep-only -o results/
        """,
    )

    parser.add_argument("input_file", type=str, nargs="?", help="Path to benchmark input file.")
    parser.add_argument("-o", "--output-dir", type=str, default="benchmark_outputs", help="Directory to save outputs.")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument("-k", "--k-min", type=int, default=5, help="Baseline k-anonymity minimum threshold (hard block).")
    parser.add_argument("--db-path", type=str, default="frequency_tables.db", help="Path to frequency table database.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint and start fresh.")
    parser.add_argument(
        "--sweep-only", action="store_true",
        help="Skip running the pipeline and regenerate analysis/sweep/plots from a saved pipeline_results.json.",
    )
    parser.add_argument("--baseline-tau-review", type=float, default=0.30, help="Baseline tau_review.")
    parser.add_argument("--baseline-tau-block", type=float, default=0.70, help="Baseline tau_block.")
    parser.add_argument("--baseline-k-lower-threshold", type=float, default=20, help="Baseline k_lower_threshold (soft review threshold on k_lower).")
    parser.add_argument("--baseline-w-direct", type=float, default=0.50, help="Baseline direct identifier weight.")
    parser.add_argument("--baseline-w-quasi", type=float, default=0.50, help="Baseline quasi identifier weight.")

    args = parser.parse_args()

    if not args.sweep_only and not args.input_file:
        parser.error("input_file is required unless --sweep-only is used.")

    run_benchmark_analysis(
        input_file=args.input_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        k_minimum=args.k_min,
        fresh_start=args.fresh,
        db_path=args.db_path,
        sweep_only=args.sweep_only,
        baseline_tau_review=args.baseline_tau_review,
        baseline_tau_block=args.baseline_tau_block,
        baseline_k_lower_threshold=args.baseline_k_lower_threshold,
        baseline_w_direct=args.baseline_w_direct,
        baseline_w_quasi=args.baseline_w_quasi,
    )


if __name__ == "__main__":
    main()