#!/usr/bin/env python3
"""
privacy_benchmark.py - Combined benchmark and analysis for privacy-preserving router.

Runs the privacy router pipeline on benchmark samples, then performs threshold
sweeps to find optimal operating points. Generates visualizations showing
effects of parameters on precision/recall and FP/FN rates.

Usage:
    python privacy_benchmark.py merged.txt
    python privacy_benchmark.py merged.txt --fresh --max-samples 50
    python privacy_benchmark.py merged.txt -o results/ -k 5
    python privacy_benchmark.py merged.txt --plot-only -o results/
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

from policy import PolicyProfile, Tier
from frequency_tables import LocalFrequencyTable
from router import PrivacyRouter, RoutingResult

# =============================================================================
# Utilities
# =============================================================================

def _write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom > 0 else default


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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
    llm_decision: Optional[str]
    llm_explanation: Optional[str]
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
    tiers_by_k: Dict[int, int] = field(default_factory=dict)
    k_lowers_by_k: Dict[int, float] = field(default_factory=dict)
    processed_at: str = ""

    def is_wikipedia(self) -> bool:
        return "wiki" in self.sample_name.lower()


@dataclass
class OperatingPoint:
    """Metrics for a single operating point configuration."""
    sweep_name: str = ""
    tau_review: float = 0.30
    tau_block: float = 0.70
    k_min: int = 5
    k_safe_threshold: int = 20
    tau_direct_review: float = 0.20
    tau_direct_block: float = 0.60
    tau_quasi_review: float = 0.30
    tau_quasi_block: float = 0.70
    w_direct: float = 0.50
    w_quasi: float = 0.50
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    marked_sensitive_rate: float = 0.0
    pass_rate: float = 0.0
    review_rate: float = 0.0
    block_rate: float = 0.0
    wiki_false_positive_rate: float = 0.0
    wiki_marked_sensitive_count: int = 0
    wiki_total: int = 0


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


# =============================================================================
# Input Parsing
# =============================================================================

def parse_input_file(file_path: str, max_samples: Optional[int] = None) -> List[BenchmarkSample]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

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
        if max_samples and len(samples) >= max_samples:
            break

    if samples:
        return samples

    # Fallback parser
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    for i, chunk in enumerate(chunks):
        samples.append(BenchmarkSample(name=f"sample_{i+1}", text=chunk, expected_critical=False))
        if max_samples and len(samples) >= max_samples:
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
            if data.get("input_hash") != self.input_hash or data.get("k_minimum") != self.k_minimum:
                print("Checkpoint mismatch. Ignoring.")
                return
            for sample_hash, raw in data.get("results", {}).items():
                raw = dict(raw)
                raw["tiers_by_k"] = {int(k): v for k, v in raw.get("tiers_by_k", {}).items()}
                raw["k_lowers_by_k"] = {int(k): v for k, v in raw.get("k_lowers_by_k", {}).items()}
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
# Pipeline Runner
# =============================================================================

class BenchmarkPipeline:
    """Runs the privacy router pipeline with checkpointing."""

    def __init__(self, checkpoint_manager: CheckpointManager, k_minimum: int = 5, db_path: str = "frequency_tables.db"):
        self.checkpoint = checkpoint_manager
        self.k_minimum = k_minimum
        self.db_path = db_path
        print("Initializing frequency table...")
        self.freq_table = LocalFrequencyTable(db_path)
        print("Initializing privacy router...")
        self._init_router()

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
        self.router = PrivacyRouter(policy=policy, frequency_table=self.freq_table)

    def _run_single_k(self, sample: BenchmarkSample, k_value: int) -> Tuple[int, float]:
        policy = PolicyProfile(
            name="benchmark_profile", version="1.0.0", reference_population="hospital_2024",
            k_minimum=k_value, k_safe_threshold=20, cloud_allows_phi=False,
            cloud_allows_masked_phi=True, require_joint_estimate=True, contextual_review_threshold=0.30,
        )
        router = PrivacyRouter(policy=policy, freq_table=self.freq_table)
        result = router.route(sample.text)
        k_lower = result.risk_estimate.lower_bound_k if result.risk_estimate and result.risk_estimate.lower_bound_k else 0.0
        return result.decision.tier.value, k_lower

    def process_sample(self, sample: BenchmarkSample) -> PipelineResult:
        start = time.time()
        result: RoutingResult = self.router.route(sample.text)
        total_time = time.time() - start

        llm_time = result.llm_time if result.llm_time else 0.0
        non_llm_time = max(total_time - llm_time, 0.0)
        detection_time = non_llm_time * 0.5
        risk_estimation_time = non_llm_time * 0.3
        gate_time = non_llm_time * 0.2

        gate_decision = result.decision.tier.name
        gate_reasons = list(result.decision.uncertainty_flags or [])
        k_lower = result.risk_estimate.lower_bound_k if result.risk_estimate and result.risk_estimate.lower_bound_k else 0.0
        k_upper = result.risk_estimate.upper_bound_k if result.risk_estimate and result.risk_estimate.upper_bound_k else float("inf")

        detected = result.detected_entities or []
        direct_identifier_count = sum(1 for e in detected if getattr(e, "is_direct_identifier", False))
        qi_count = len(detected) - direct_identifier_count

        joint_risk = 0.0 if not k_lower or k_lower == float("inf") else min(1.0, 1.0 / k_lower)

        original_len = len(sample.text)
        masked_len = len(result.masked_text) if result.masked_text else original_len
        masking_ratio = (original_len - masked_len) / original_len if original_len > 0 else 0.0

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

        return PipelineResult(
            sample_name=sample.name, sample_hash=sample.get_hash(), text_length=len(sample.text),
            expected_critical=sample.expected_critical, tier=result.decision.tier.value,
            gate_decision=gate_decision, gate_reasons=gate_reasons, k_lower=k_lower, k_upper=k_upper,
            joint_risk_score=joint_risk, direct_identifier_count=direct_identifier_count,
            quasi_identifier_count=qi_count, llm_invoked=result.decision.contextual_review_invoked,
            llm_decision=None, llm_explanation=None, original_text_length=original_len,
            masked_text_length=masked_len, masking_ratio=masking_ratio,
            hard_stop_triggered=bool(result.decision.hard_stop_reasons),
            hard_stop_reasons=result.decision.hard_stop_reasons,
            uncertainty_flags=result.decision.uncertainty_flags, total_time=total_time,
            detection_time=detection_time, risk_estimation_time=risk_estimation_time,
            gate_time=gate_time, llm_time=llm_time, tiers_by_k=tiers_by_k, k_lowers_by_k=k_lowers_by_k,
            processed_at=datetime.now().isoformat(),
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

            print(f"[{i+1}/{total}] Processing {sample.name}")
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
        print(f"Processed new: {processed_new}, Skipped existing: {skipped}")

        return [self.checkpoint.get_result(s.get_hash()) for s in samples if self.checkpoint.get_result(s.get_hash())]


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_results(results: List[PipelineResult]) -> BenchmarkAnalysis:
    total = len(results)
    tier_1 = [r for r in results if r.tier == 1]
    tier_2 = [r for r in results if r.tier == 2]
    tier_3 = [r for r in results if r.tier == 3]
    false_cloud = [r for r in results if r.tier == 1 and r.expected_critical]
    over_censor = [r for r in results if r.tier == 3 and not r.expected_critical]
    llm_invoked = [r for r in results if r.llm_invoked]

    k_lowers = [r.k_lower for r in results if r.k_lower and np.isfinite(r.k_lower)]
    expected = np.array([1 if r.expected_critical else 0 for r in results])

    correlations = {
        "joint_risk_score": safe_corr(expected, np.array([r.joint_risk_score for r in results])),
        "k_lower": safe_corr(expected, np.array([r.k_lower if r.k_lower and np.isfinite(r.k_lower) else 0.0 for r in results])),
        "direct_identifier_count": safe_corr(expected, np.array([r.direct_identifier_count for r in results])),
        "quasi_identifier_count": safe_corr(expected, np.array([r.quasi_identifier_count for r in results])),
    }

    return BenchmarkAnalysis(
        total_samples=total,
        tier_1_count=len(tier_1), tier_2_count=len(tier_2), tier_3_count=len(tier_3),
        tier_1_pct=safe_div(len(tier_1), total) * 100,
        tier_2_pct=safe_div(len(tier_2), total) * 100,
        tier_3_pct=safe_div(len(tier_3), total) * 100,
        false_cloud_release_count=len(false_cloud),
        false_cloud_release_rate=safe_div(len(false_cloud), total) * 100,
        over_censoring_count=len(over_censor),
        over_censoring_rate=safe_div(len(over_censor), total) * 100,
        llm_invocation_count=len(llm_invoked),
        llm_invocation_rate=safe_div(len(llm_invoked), total) * 100,
        mean_k_lower=float(np.mean(k_lowers)) if k_lowers else 0.0,
        median_k_lower=float(np.median(k_lowers)) if k_lowers else 0.0,
        mean_k_lower_by_tier={
            "tier_1": float(np.mean([r.k_lower for r in tier_1 if r.k_lower and np.isfinite(r.k_lower)])) if tier_1 else 0.0,
            "tier_2": float(np.mean([r.k_lower for r in tier_2 if r.k_lower and np.isfinite(r.k_lower)])) if tier_2 else 0.0,
            "tier_3": float(np.mean([r.k_lower for r in tier_3 if r.k_lower and np.isfinite(r.k_lower)])) if tier_3 else 0.0,
        },
        mean_total_time=float(np.mean([r.total_time for r in results])),
        mean_llm_time=float(np.mean([r.llm_time for r in results])),
        correlations=correlations,
    )


# =============================================================================
# Sweep Simulation
# =============================================================================

def compute_normalization(results: List[PipelineResult]) -> Tuple[float, float]:
    max_direct = max((r.direct_identifier_count for r in results), default=1) or 1
    max_quasi = max((r.quasi_identifier_count for r in results), default=1) or 1
    return float(max_direct), float(max_quasi)


def simulate_routing(
    result: PipelineResult,
    max_direct: float,
    max_quasi: float,
    tau_review: float,
    tau_block: float,
    k_min: int,
    k_safe_threshold: int,
    tau_direct_review: float,
    tau_direct_block: float,
    tau_quasi_review: float,
    tau_quasi_block: float,
    w_direct: float,
    w_quasi: float,
) -> Tuple[bool, bool, bool]:
    """Simulate routing decision. Returns: (marked_sensitive, is_review, is_block)"""
    direct_score = result.direct_identifier_count / max_direct
    quasi_score = result.quasi_identifier_count / max_quasi
    combined_risk = (result.joint_risk_score + w_direct * direct_score + w_quasi * quasi_score) / (1.0 + w_direct + w_quasi)

    # Block conditions
    if result.k_lower < k_min:
        return True, False, True
    if direct_score >= tau_direct_block:
        return True, False, True
    if quasi_score >= tau_quasi_block:
        return True, False, True
    if combined_risk >= tau_block:
        return True, False, True

    # Review conditions
    if direct_score >= tau_direct_review:
        return True, True, False
    if quasi_score >= tau_quasi_review:
        return True, True, False
    if combined_risk >= tau_review:
        return True, True, False
    if result.k_lower < k_safe_threshold:
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
    k_safe_threshold: int,
    tau_direct_review: float,
    tau_direct_block: float,
    tau_quasi_review: float,
    tau_quasi_block: float,
    w_direct: float,
    w_quasi: float,
) -> OperatingPoint:
    tp = fp = tn = fn = 0
    review_count = block_count = 0
    wiki_total = wiki_marked = 0

    for r in results:
        marked, is_review, is_block = simulate_routing(
            r, max_direct, max_quasi, tau_review, tau_block, k_min, k_safe_threshold,
            tau_direct_review, tau_direct_block, tau_quasi_review, tau_quasi_block, w_direct, w_quasi
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

        if r.is_wikipedia():
            wiki_total += 1
            if marked:
                wiki_marked += 1

    total = len(results)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    return OperatingPoint(
        sweep_name=sweep_name, tau_review=tau_review, tau_block=tau_block, k_min=k_min,
        k_safe_threshold=k_safe_threshold, tau_direct_review=tau_direct_review,
        tau_direct_block=tau_direct_block, tau_quasi_review=tau_quasi_review,
        tau_quasi_block=tau_quasi_block, w_direct=w_direct, w_quasi=w_quasi,
        precision=precision, recall=recall, f1=f1, specificity=specificity,
        false_positive_rate=fpr, false_negative_rate=fnr,
        accuracy=safe_div(tp + tn, total), balanced_accuracy=(recall + specificity) / 2,
        true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
        marked_sensitive_rate=safe_div(tp + fp, total), pass_rate=safe_div(tn + fn, total),
        review_rate=safe_div(review_count, total), block_rate=safe_div(block_count, total),
        wiki_false_positive_rate=safe_div(wiki_marked, wiki_total),
        wiki_marked_sensitive_count=wiki_marked, wiki_total=wiki_total,
    )


def run_sweeps(
    results: List[PipelineResult],
    baseline_tau_review: float = 0.30,
    baseline_tau_block: float = 0.70,
    baseline_k_min: int = 5,
    baseline_k_safe_threshold: int = 20,
    baseline_tau_direct_review: float = 0.20,
    baseline_tau_direct_block: float = 0.60,
    baseline_tau_quasi_review: float = 0.30,
    baseline_tau_quasi_block: float = 0.70,
    baseline_w_direct: float = 0.50,
    baseline_w_quasi: float = 0.50,
) -> Dict[str, List[OperatingPoint]]:
    """Run all threshold/weight sweeps."""
    max_direct, max_quasi = compute_normalization(results)
    sweeps: Dict[str, List[OperatingPoint]] = {}

    # Define sweep ranges
    tau_review_values = [round(x, 2) for x in np.arange(0.05, 0.55, 0.02)]
    tau_block_values = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]
    k_min_values = list(range(2, 21))
    k_safe_threshold_values = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 1000, 10000]
    tau_direct_review_values = [round(x, 2) for x in np.arange(0.05, 0.50, 0.05)]
    tau_direct_block_values = [round(x, 2) for x in np.arange(0.40, 0.95, 0.05)]
    tau_quasi_review_values = [round(x, 2) for x in np.arange(0.10, 0.55, 0.05)]
    tau_quasi_block_values = [round(x, 2) for x in np.arange(0.50, 0.95, 0.05)]
    weight_values = [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)]

    def eval_point(name: str, **kwargs) -> OperatingPoint:
        defaults = dict(
            tau_review=baseline_tau_review, tau_block=baseline_tau_block, k_min=baseline_k_min,
            k_safe_threshold=baseline_k_safe_threshold, tau_direct_review=baseline_tau_direct_review,
            tau_direct_block=baseline_tau_direct_block, tau_quasi_review=baseline_tau_quasi_review,
            tau_quasi_block=baseline_tau_quasi_block, w_direct=baseline_w_direct, w_quasi=baseline_w_quasi,
        )
        defaults.update(kwargs)
        return evaluate_operating_point(results, max_direct, max_quasi, name, **defaults)

    print("  Running tau_review sweep...")
    sweeps["tau_review_sweep"] = [eval_point("tau_review_sweep", tau_review=v) for v in tau_review_values]

    print("  Running tau_block sweep...")
    sweeps["tau_block_sweep"] = [eval_point("tau_block_sweep", tau_block=v) for v in tau_block_values]

    print("  Running k_min sweep...")
    sweeps["k_min_sweep"] = [eval_point("k_min_sweep", k_min=v) for v in k_min_values]

    print("  Running k_safe_threshold sweep...")
    sweeps["k_safe_threshold_sweep"] = [eval_point("k_safe_threshold_sweep", k_safe_threshold=v) for v in k_safe_threshold_values]

    print("  Running tau_direct_review sweep...")
    sweeps["tau_direct_review_sweep"] = [eval_point("tau_direct_review_sweep", tau_direct_review=v) for v in tau_direct_review_values]

    print("  Running tau_direct_block sweep...")
    sweeps["tau_direct_block_sweep"] = [eval_point("tau_direct_block_sweep", tau_direct_block=v) for v in tau_direct_block_values]

    print("  Running tau_quasi_review sweep...")
    sweeps["tau_quasi_review_sweep"] = [eval_point("tau_quasi_review_sweep", tau_quasi_review=v) for v in tau_quasi_review_values]

    print("  Running tau_quasi_block sweep...")
    sweeps["tau_quasi_block_sweep"] = [eval_point("tau_quasi_block_sweep", tau_quasi_block=v) for v in tau_quasi_block_values]

    print("  Running w_direct sweep...")
    sweeps["w_direct_sweep"] = [eval_point("w_direct_sweep", w_direct=v) for v in weight_values]

    print("  Running w_quasi sweep...")
    sweeps["w_quasi_sweep"] = [eval_point("w_quasi_sweep", w_quasi=v) for v in weight_values]

    return sweeps


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(results: List[PipelineResult], output_dir: Path, **baseline_kwargs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running sweeps for visualizations...")
    sweeps = run_sweeps(results, **baseline_kwargs)

    # Save sweep results as JSON
    sweep_data = {name: [asdict(p) for p in points] for name, points in sweeps.items()}
    _write_json(output_dir / "sweep_results.json", sweep_data)

    # Plot precision/recall/FPR/FNR for each sweep
    for sweep_name, points in sweeps.items():
        if not points:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Get x-axis values based on sweep type
        if "tau_review" in sweep_name and "direct" not in sweep_name and "quasi" not in sweep_name:
            x = [p.tau_review for p in points]
            xlabel = "tau_review"
        elif "tau_block" in sweep_name and "direct" not in sweep_name and "quasi" not in sweep_name:
            x = [p.tau_block for p in points]
            xlabel = "tau_block"
        elif sweep_name == "k_min_sweep":
            x = [p.k_min for p in points]
            xlabel = "k_min"
        elif sweep_name == "k_safe_threshold_sweep":
            x = [p.k_safe_threshold for p in points]
            xlabel = "k_safe_threshold"
        elif "tau_direct_review" in sweep_name:
            x = [p.tau_direct_review for p in points]
            xlabel = "tau_direct_review"
        elif "tau_direct_block" in sweep_name:
            x = [p.tau_direct_block for p in points]
            xlabel = "tau_direct_block"
        elif "tau_quasi_review" in sweep_name:
            x = [p.tau_quasi_review for p in points]
            xlabel = "tau_quasi_review"
        elif "tau_quasi_block" in sweep_name:
            x = [p.tau_quasi_block for p in points]
            xlabel = "tau_quasi_block"
        elif "w_direct" in sweep_name:
            x = [p.w_direct for p in points]
            xlabel = "w_direct"
        elif "w_quasi" in sweep_name:
            x = [p.w_quasi for p in points]
            xlabel = "w_quasi"
        else:
            continue

        # Precision/Recall plot
        ax = axes[0]
        ax.plot(x, [p.precision for p in points], "b-o", label="Precision")
        ax.plot(x, [p.recall for p in points], "g-s", label="Recall")
        ax.plot(x, [p.f1 for p in points], "m-^", label="F1")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Score")
        ax.set_title(f"{sweep_name}: Precision/Recall")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

        # FPR/FNR plot
        ax = axes[1]
        ax.plot(x, [p.false_positive_rate for p in points], "r-o", label="FPR")
        ax.plot(x, [p.false_negative_rate for p in points], "orange", marker="s", label="FNR")
        ax.plot(x, [p.wiki_false_positive_rate for p in points], "c-^", label="Wiki FPR")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Rate")
        ax.set_title(f"{sweep_name}: FPR/FNR")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

        fig.tight_layout()
        fig.savefig(output_dir / f"{sweep_name}.png", dpi=150)
        plt.close(fig)

    # Create combined comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Direct thresholds comparison
    ax = axes[0, 0]
    for sweep_name in ["tau_direct_review_sweep", "tau_direct_block_sweep"]:
        points = sweeps.get(sweep_name, [])
        if points:
            if "review" in sweep_name:
                x = [p.tau_direct_review for p in points]
                label = "Direct Review"
            else:
                x = [p.tau_direct_block for p in points]
                label = "Direct Block"
            ax.plot(x, [p.recall for p in points], "-o", label=f"{label} - Recall")
            ax.plot(x, [p.false_positive_rate for p in points], "--s", label=f"{label} - FPR")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Direct Identifier Thresholds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Quasi thresholds comparison
    ax = axes[0, 1]
    for sweep_name in ["tau_quasi_review_sweep", "tau_quasi_block_sweep"]:
        points = sweeps.get(sweep_name, [])
        if points:
            if "review" in sweep_name:
                x = [p.tau_quasi_review for p in points]
                label = "Quasi Review"
            else:
                x = [p.tau_quasi_block for p in points]
                label = "Quasi Block"
            ax.plot(x, [p.recall for p in points], "-o", label=f"{label} - Recall")
            ax.plot(x, [p.false_positive_rate for p in points], "--s", label=f"{label} - FPR")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Quasi Identifier Thresholds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # k thresholds comparison
    ax = axes[1, 0]
    for sweep_name in ["k_min_sweep", "k_safe_threshold_sweep"]:
        points = sweeps.get(sweep_name, [])
        if points:
            if "k_min" in sweep_name:
                x = [p.k_min for p in points]
                label = "k_min"
            else:
                x = [p.k_safe_threshold for p in points]
                label = "k_safe_threshold"
            ax.plot(x, [p.recall for p in points], "-o", label=f"{label} - Recall")
            ax.plot(x, [p.wiki_false_positive_rate for p in points], "--^", label=f"{label} - Wiki FPR")
    ax.set_xlabel("k value")
    ax.set_ylabel("Rate")
    ax.set_title("k-Anonymity Thresholds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Weight comparison
    ax = axes[1, 1]
    for sweep_name in ["w_direct_sweep", "w_quasi_sweep"]:
        points = sweeps.get(sweep_name, [])
        if points:
            if "w_direct" in sweep_name:
                x = [p.w_direct for p in points]
                label = "w_direct"
            else:
                x = [p.w_quasi for p in points]
                label = "w_quasi"
            ax.plot(x, [p.f1 for p in points], "-o", label=f"{label} - F1")
    ax.set_xlabel("Weight")
    ax.set_ylabel("F1 Score")
    ax.set_title("Weight Parameters")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Parameter Sweep Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "parameter_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Saved visualizations to {output_dir}")


# =============================================================================
# Load Results
# =============================================================================

def load_results_from_output_dir(output_dir: Path) -> List[PipelineResult]:
    path = output_dir / "pipeline_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No pipeline_results.json found at {path}")

    raw = _read_json(path)
    results = []
    for item in raw:
        item = dict(item)
        item["tiers_by_k"] = {int(k): v for k, v in item.get("tiers_by_k", {}).items()}
        item["k_lowers_by_k"] = {int(k): v for k, v in item.get("k_lowers_by_k", {}).items()}
        results.append(PipelineResult(**item))
    return results


# =============================================================================
# Main Runner
# =============================================================================

def run_benchmark(
    input_file: str,
    output_dir: str = "benchmark_outputs",
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
    print("PRIVACY ROUTER BENCHMARK")
    print("=" * 70)

    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    print(f"Loading samples from: {input_file}")
    samples = parse_input_file(input_file, max_samples)
    print(f"Loaded {len(samples)} benchmark samples")
    print(f"  Expected critical: {sum(1 for s in samples if s.expected_critical)}")
    print(f"  Expected non-critical: {sum(1 for s in samples if not s.expected_critical)}")

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

    if results and not checkpoint.was_interrupted:
        analysis = analyze_results(results)
        _write_json(out_path / "analysis.json", asdict(analysis))

        print("\n" + "-" * 70)
        print("ANALYSIS")
        print("-" * 70)
        print(f"Total samples: {analysis.total_samples}")
        print(f"Tier distribution: T1={analysis.tier_1_pct:.1f}%, T2={analysis.tier_2_pct:.1f}%, T3={analysis.tier_3_pct:.1f}%")
        print(f"False cloud release rate: {analysis.false_cloud_release_rate:.2f}%")
        print(f"Over-censoring rate: {analysis.over_censoring_rate:.2f}%")

        print("\nGenerating visualizations...")
        create_visualizations(
            results=results,
            output_dir=out_path,
            baseline_tau_review=baseline_tau_review,
            baseline_tau_block=baseline_tau_block,
            baseline_k_min=k_minimum,
            baseline_k_safe_threshold=20,
            baseline_tau_direct_review=0.20,
            baseline_tau_direct_block=0.60,
            baseline_tau_quasi_review=0.30,
            baseline_tau_quasi_block=0.70,
            baseline_w_direct=baseline_w_direct,
            baseline_w_quasi=baseline_w_quasi,
        )

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
        description="Benchmark privacy-preserving router with threshold sweep analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python privacy_benchmark.py merged.txt
  python privacy_benchmark.py merged.txt --fresh --max-samples 50
  python privacy_benchmark.py merged.txt -o results/ -k 5
  python privacy_benchmark.py merged.txt --plot-only -o results/
        """,
    )

    parser.add_argument("input_file", type=str, nargs="?", help="Path to benchmark input file.")
    parser.add_argument("-o", "--output-dir", type=str, default="benchmark_outputs", help="Directory to save outputs.")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument("-k", "--k-minimum", type=int, default=5, help="k-anonymity minimum threshold.")
    parser.add_argument("--db-path", type=str, default="frequency_tables.db", help="Path to frequency table database.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint and start fresh.")
    parser.add_argument("--plot-only", action="store_true", help="Skip benchmark, regenerate plots from saved results.")
    parser.add_argument("--baseline-tau-review", type=float, default=0.30, help="Baseline tau_review.")
    parser.add_argument("--baseline-tau-block", type=float, default=0.70, help="Baseline tau_block.")
    parser.add_argument("--baseline-w-direct", type=float, default=0.50, help="Baseline direct identifier weight.")
    parser.add_argument("--baseline-w-quasi", type=float, default=0.50, help="Baseline quasi identifier weight.")

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
            baseline_k_min=args.k_minimum,
            baseline_k_safe_threshold=20,
            baseline_tau_direct_review=0.20,
            baseline_tau_direct_block=0.60,
            baseline_tau_quasi_review=0.30,
            baseline_tau_quasi_block=0.70,
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