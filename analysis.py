#!/usr/bin/env python3
"""
analysis.py - Threshold sensitivity analysis for privacy routing benchmarks.

This script analyzes saved benchmark results from pipeline_results.json and
simulates how different routing thresholds affect privacy, recall, precision,
false positives, and routing tier behavior.

It does not rerun the router. It reuses existing benchmark outputs.

Expected input:
    pipeline_results.json

Example usage:
    python analysis.py -i benchmark_revised_outputs/pipeline_results.json -o analysis_outputs

Outputs:
    analysis_outputs/
        threshold_sweep_results.json
        analysis_summary.json
        latex_writeup.tex
        01_threshold_phase_diagram.png
        02_precision_recall_curves.png
        03_privacy_false_positive_tradeoff.png
        04_k_minimum_sensitivity.png
        05_weight_sensitivity_heatmaps.png
        06_operating_point_breakdown.png
        07_wikipedia_false_positive_analysis.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    sample_name: str
    sample_hash: str
    expected_critical: bool

    tier: Optional[int]
    joint_risk_score: float
    k_lower: Optional[float]

    direct_identifier_count: int
    quasi_identifier_count: int
    llm_invoked: bool

    text_length: Optional[int] = None
    original_text_length: Optional[int] = None
    masked_text_length: Optional[int] = None
    masking_ratio: Optional[float] = None

    gate_decision: Optional[str] = None


@dataclass
class OperatingPoint:
    tau_review: float
    tau_block: float
    k_min: int
    w_direct: float
    w_quasi: float

    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    specificity: float
    accuracy: float
    balanced_accuracy: float

    marked_sensitive_rate: float
    pass_rate: float
    review_rate: float
    block_rate: float

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    total: int
    positives: int
    negatives: int

    wiki_false_positive_rate: Optional[float] = None
    wiki_marked_sensitive_count: Optional[int] = None
    wiki_total: Optional[int] = None


@dataclass
class RoutingCounts:
    passed: int = 0
    review: int = 0
    block: int = 0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0:
        return default
    return num / den


def frange(start: float, stop: float, step: float) -> List[float]:
    """
    Inclusive floating-point range with rounding for clean labels.
    """
    values = []
    x = start
    while x <= stop + 1e-12:
        values.append(round(x, 10))
        x += step
    return values


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def is_wikipedia_sample(sample: SampleResult) -> bool:
    name = sample.sample_name.lower()
    return "wiki" in name or "wikipedia" in name


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Loading benchmark results
# ---------------------------------------------------------------------------

def parse_sample(raw: Dict[str, Any]) -> SampleResult:
    return SampleResult(
        sample_name=str(raw.get("sample_name", "")),
        sample_hash=str(raw.get("sample_hash", "")),
        expected_critical=bool(raw.get("expected_critical", False)),

        tier=raw.get("tier"),
        joint_risk_score=float(raw.get("joint_risk_score") or 0.0),
        k_lower=raw.get("k_lower"),

        direct_identifier_count=int(raw.get("direct_identifier_count") or 0),
        quasi_identifier_count=int(raw.get("quasi_identifier_count") or 0),
        llm_invoked=bool(raw.get("llm_invoked", False)),

        text_length=raw.get("text_length"),
        original_text_length=raw.get("original_text_length"),
        masked_text_length=raw.get("masked_text_length"),
        masking_ratio=raw.get("masking_ratio"),

        gate_decision=raw.get("gate_decision"),
    )


def load_samples(path: Path) -> List[SampleResult]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError("Expected pipeline_results.json to contain a list of result objects.")
    return [parse_sample(x) for x in raw]


# ---------------------------------------------------------------------------
# Risk simulation model
# ---------------------------------------------------------------------------

def normalized_feature_counts(samples: List[SampleResult]) -> Tuple[float, float]:
    """
    Returns max direct and quasi counts for normalization.

    We normalize counts by observed maxima so that w_direct and w_quasi
    contribute comparable values to the adjusted risk.
    """
    max_direct = max((s.direct_identifier_count for s in samples), default=1)
    max_quasi = max((s.quasi_identifier_count for s in samples), default=1)

    return max(float(max_direct), 1.0), max(float(max_quasi), 1.0)


def adjusted_risk(
    sample: SampleResult,
    max_direct: float,
    max_quasi: float,
    w_direct: float,
    w_quasi: float,
) -> float:
    """
    Computes a simulated risk score.

    The saved benchmark already contains joint_risk_score. This adds explicit
    sensitivity to direct and quasi identifier counts so we can study how
    feature weights affect the threshold behavior.

    adjusted_risk =
        joint_risk_score
        + w_direct * normalized_direct_identifier_count
        + w_quasi  * normalized_quasi_identifier_count

    The result is clipped to [0, 1].
    """
    direct_component = safe_div(sample.direct_identifier_count, max_direct)
    quasi_component = safe_div(sample.quasi_identifier_count, max_quasi)

    risk = (
        sample.joint_risk_score
        + w_direct * direct_component
        + w_quasi * quasi_component
    )

    return clamp01(risk)


def simulated_route(
    risk: float,
    k_lower: Optional[float],
    tau_review: float,
    tau_block: float,
    k_min: int,
) -> str:
    """
    Simulates a three-way routing decision.

    pass:
        sample is treated as safe enough for cloud processing.

    review:
        sample is sensitive enough to require local LLM/contextual review.

    block:
        sample is considered too sensitive for cloud routing.

    Rule:
        if k_lower exists and k_lower < k_min:
            block
        elif risk >= tau_block:
            block
        elif risk >= tau_review:
            review
        else:
            pass
    """
    if k_lower is not None:
        try:
            if float(k_lower) < float(k_min):
                return "block"
        except (TypeError, ValueError):
            pass

    if risk >= tau_block:
        return "block"

    if risk >= tau_review:
        return "review"

    return "pass"


def is_marked_sensitive(route: str) -> bool:
    """
    For classification metrics, both review and block count as positive.

    Positive prediction:
        "This text needs local/sensitive handling."

    Negative prediction:
        "This text may pass."
    """
    return route in {"review", "block"}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def evaluate_operating_point(
    samples: List[SampleResult],
    tau_review: float,
    tau_block: float,
    k_min: int,
    w_direct: float,
    w_quasi: float,
) -> OperatingPoint:
    max_direct, max_quasi = normalized_feature_counts(samples)

    tp = fp = tn = fn = 0
    routing = RoutingCounts()

    wiki_total = 0
    wiki_marked_sensitive = 0

    for sample in samples:
        risk = adjusted_risk(
            sample=sample,
            max_direct=max_direct,
            max_quasi=max_quasi,
            w_direct=w_direct,
            w_quasi=w_quasi,
        )

        route = simulated_route(
            risk=risk,
            k_lower=sample.k_lower,
            tau_review=tau_review,
            tau_block=tau_block,
            k_min=k_min,
        )

        if route == "pass":
            routing.passed += 1
        elif route == "review":
            routing.review += 1
        elif route == "block":
            routing.block += 1

        predicted_positive = is_marked_sensitive(route)
        actual_positive = sample.expected_critical

        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and not actual_positive:
            tn += 1
        elif not predicted_positive and actual_positive:
            fn += 1

        if is_wikipedia_sample(sample):
            wiki_total += 1
            if predicted_positive:
                wiki_marked_sensitive += 1

    total = len(samples)
    positives = tp + fn
    negatives = tn + fp

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_positive_rate = safe_div(fp, fp + tn)
    false_negative_rate = safe_div(fn, fn + tp)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, total)
    balanced_accuracy = 0.5 * (recall + specificity)

    marked_sensitive_rate = safe_div(tp + fp, total)
    pass_rate = safe_div(routing.passed, total)
    review_rate = safe_div(routing.review, total)
    block_rate = safe_div(routing.block, total)

    wiki_fpr = None
    if wiki_total > 0:
        wiki_fpr = safe_div(wiki_marked_sensitive, wiki_total)

    return OperatingPoint(
        tau_review=tau_review,
        tau_block=tau_block,
        k_min=k_min,
        w_direct=w_direct,
        w_quasi=w_quasi,

        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        specificity=specificity,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,

        marked_sensitive_rate=marked_sensitive_rate,
        pass_rate=pass_rate,
        review_rate=review_rate,
        block_rate=block_rate,

        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,

        total=total,
        positives=positives,
        negatives=negatives,

        wiki_false_positive_rate=wiki_fpr,
        wiki_marked_sensitive_count=wiki_marked_sensitive,
        wiki_total=wiki_total,
    )


def run_sweeps(
    samples: List[SampleResult],
    tau_review_values: List[float],
    tau_block_values: List[float],
    k_min_values: List[int],
    weight_values: List[float],
    baseline_tau_review: float,
    baseline_tau_block: float,
    baseline_k_min: int,
    baseline_w_direct: float,
    baseline_w_quasi: float,
) -> Dict[str, List[OperatingPoint]]:
    """
    Runs targeted sweeps.

    Instead of exhaustively plotting every possible combination, this creates
    interpretable families of experiments:

    1. tau_review sweep:
        varies review threshold, holds others fixed.

    2. tau_block sweep:
        varies block threshold, holds others fixed.

    3. k_min sweep:
        varies k-anonymity floor, holds others fixed.

    4. direct/quasi weight grid:
        varies w_direct and w_quasi, holds thresholds fixed.

    5. tau_review/tau_block grid:
        varies both thresholds, holds k and weights fixed.
    """
    results: Dict[str, List[OperatingPoint]] = {
        "tau_review_sweep": [],
        "tau_block_sweep": [],
        "k_min_sweep": [],
        "weight_grid": [],
        "threshold_grid": [],
    }

    for tau_review in tau_review_values:
        if tau_review < baseline_tau_block:
            results["tau_review_sweep"].append(
                evaluate_operating_point(
                    samples=samples,
                    tau_review=tau_review,
                    tau_block=baseline_tau_block,
                    k_min=baseline_k_min,
                    w_direct=baseline_w_direct,
                    w_quasi=baseline_w_quasi,
                )
            )

    for tau_block in tau_block_values:
        if baseline_tau_review < tau_block:
            results["tau_block_sweep"].append(
                evaluate_operating_point(
                    samples=samples,
                    tau_review=baseline_tau_review,
                    tau_block=tau_block,
                    k_min=baseline_k_min,
                    w_direct=baseline_w_direct,
                    w_quasi=baseline_w_quasi,
                )
            )

    for k_min in k_min_values:
        results["k_min_sweep"].append(
            evaluate_operating_point(
                samples=samples,
                tau_review=baseline_tau_review,
                tau_block=baseline_tau_block,
                k_min=k_min,
                w_direct=baseline_w_direct,
                w_quasi=baseline_w_quasi,
            )
        )

    for w_direct in weight_values:
        for w_quasi in weight_values:
            results["weight_grid"].append(
                evaluate_operating_point(
                    samples=samples,
                    tau_review=baseline_tau_review,
                    tau_block=baseline_tau_block,
                    k_min=baseline_k_min,
                    w_direct=w_direct,
                    w_quasi=w_quasi,
                )
            )

    for tau_review in tau_review_values:
        for tau_block in tau_block_values:
            if tau_review < tau_block:
                results["threshold_grid"].append(
                    evaluate_operating_point(
                        samples=samples,
                        tau_review=tau_review,
                        tau_block=tau_block,
                        k_min=baseline_k_min,
                        w_direct=baseline_w_direct,
                        w_quasi=baseline_w_quasi,
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Summary selection
# ---------------------------------------------------------------------------

def pareto_candidates(points: List[OperatingPoint]) -> List[OperatingPoint]:
    """
    Returns points that are not dominated in privacy/recall and false positive rate.

    A point is dominated if another point has:
        recall >= this recall
        false_positive_rate <= this false_positive_rate
    and strictly improves at least one.
    """
    candidates = []

    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue

            better_or_equal_recall = q.recall >= p.recall
            better_or_equal_fpr = q.false_positive_rate <= p.false_positive_rate
            strictly_better = q.recall > p.recall or q.false_positive_rate < p.false_positive_rate

            if better_or_equal_recall and better_or_equal_fpr and strictly_better:
                dominated = True
                break

        if not dominated:
            candidates.append(p)

    return candidates


def choose_operating_points(points: List[OperatingPoint]) -> Dict[str, Optional[OperatingPoint]]:
    if not points:
        return {
            "best_f1": None,
            "best_balanced_accuracy": None,
            "high_recall_low_fpr": None,
            "lowest_wiki_false_positive": None,
        }

    best_f1 = max(points, key=lambda p: p.f1)
    best_balanced = max(points, key=lambda p: p.balanced_accuracy)

    # Prioritize recall >= 0.95 where possible, then minimize FPR.
    high_recall_candidates = [p for p in points if p.recall >= 0.95]
    if high_recall_candidates:
        high_recall_low_fpr = min(
            high_recall_candidates,
            key=lambda p: (p.false_positive_rate, -p.precision, -p.f1),
        )
    else:
        high_recall_low_fpr = max(points, key=lambda p: (p.recall, -p.false_positive_rate))

    wiki_candidates = [p for p in points if p.wiki_false_positive_rate is not None]
    if wiki_candidates:
        lowest_wiki_fp = min(
            wiki_candidates,
            key=lambda p: (
                p.wiki_false_positive_rate,
                -p.recall,
                p.false_positive_rate,
            ),
        )
    else:
        lowest_wiki_fp = None

    return {
        "best_f1": best_f1,
        "best_balanced_accuracy": best_balanced,
        "high_recall_low_fpr": high_recall_low_fpr,
        "lowest_wiki_false_positive": lowest_wiki_fp,
    }


def summarize_samples(samples: List[SampleResult]) -> Dict[str, Any]:
    total = len(samples)
    positives = sum(1 for s in samples if s.expected_critical)
    negatives = total - positives

    wiki_samples = [s for s in samples if is_wikipedia_sample(s)]
    wiki_total = len(wiki_samples)
    wiki_critical = sum(1 for s in wiki_samples if s.expected_critical)

    return {
        "total_samples": total,
        "expected_critical": positives,
        "expected_noncritical": negatives,
        "positive_rate": safe_div(positives, total),
        "wikipedia_samples": wiki_total,
        "wikipedia_expected_critical": wiki_critical,
        "wikipedia_expected_noncritical": wiki_total - wiki_critical,
        "mean_joint_risk_score": float(np.mean([s.joint_risk_score for s in samples])) if samples else 0.0,
        "mean_direct_identifier_count": float(np.mean([s.direct_identifier_count for s in samples])) if samples else 0.0,
        "mean_quasi_identifier_count": float(np.mean([s.quasi_identifier_count for s in samples])) if samples else 0.0,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def point_list_to_arrays(points: List[OperatingPoint], x_field: str) -> Dict[str, np.ndarray]:
    return {
        "x": np.array([getattr(p, x_field) for p in points], dtype=float),
        "precision": np.array([p.precision for p in points], dtype=float),
        "recall": np.array([p.recall for p in points], dtype=float),
        "f1": np.array([p.f1 for p in points], dtype=float),
        "fpr": np.array([p.false_positive_rate for p in points], dtype=float),
        "marked": np.array([p.marked_sensitive_rate for p in points], dtype=float),
        "pass": np.array([p.pass_rate for p in points], dtype=float),
        "review": np.array([p.review_rate for p in points], dtype=float),
        "block": np.array([p.block_rate for p in points], dtype=float),
    }


def savefig(fig, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 1: Threshold phase diagram
# ---------------------------------------------------------------------------

def plot_threshold_phase_diagram(points: List[OperatingPoint], output_dir: Path) -> None:
    """
    Shows how tau_review and tau_block jointly affect recall, FPR, FNR,
    review rate, and block rate.
    """
    if not points:
        return

    plt = ensure_matplotlib()

    tau_reviews = sorted(set(p.tau_review for p in points))
    tau_blocks = sorted(set(p.tau_block for p in points))

    def make_grid(metric: str) -> np.ndarray:
        grid = np.full((len(tau_reviews), len(tau_blocks)), np.nan)
        index_r = {v: i for i, v in enumerate(tau_reviews)}
        index_b = {v: i for i, v in enumerate(tau_blocks)}
        for p in points:
            grid[index_r[p.tau_review], index_b[p.tau_block]] = getattr(p, metric)
        return grid

    metrics = [
        ("recall", "Privacy Recall"),
        ("false_negative_rate", "False Negative Rate"),
        ("false_positive_rate", "False Positive Rate"),
        ("review_rate", "Review Rate"),
        ("block_rate", "Block Rate"),
        ("marked_sensitive_rate", "Marked Sensitive Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        grid = make_grid(metric)

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[
                min(tau_blocks),
                max(tau_blocks),
                min(tau_reviews),
                max(tau_reviews),
            ],
            vmin=0,
            vmax=1,
            cmap="viridis",
        )

        ax.set_title(title)
        ax.set_xlabel(r"$\tau_{\mathrm{block}}$")
        ax.set_ylabel(r"$\tau_{\mathrm{review}}$")

        # Invalid/unused boundary: tau_review = tau_block.
        low = max(min(tau_reviews), min(tau_blocks))
        high = min(max(tau_reviews), max(tau_blocks))
        ax.plot([low, high], [low, high], color="white", linestyle="--", linewidth=1.5)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(title)

    fig.suptitle(
        "Joint Threshold Effects: Review Threshold vs. Block Threshold",
        fontsize=16,
    )

    savefig(fig, output_dir, "01_threshold_phase_diagram.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Precision-recall curves
# ---------------------------------------------------------------------------

def plot_precision_recall_curves(sweeps: Dict[str, List[OperatingPoint]], output_dir: Path) -> None:
    """
    Plots precision-recall relationships for different threshold families.
    """
    plt = ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(9, 7))

    families = [
        ("tau_review_sweep", "vary review threshold", "o"),
        ("tau_block_sweep", "vary block threshold", "s"),
        ("k_min_sweep", "vary k_min", "^"),
        ("weight_grid", "vary direct/quasi weights", "."),
        ("threshold_grid", "vary both thresholds", "x"),
    ]

    for key, label, marker in families:
        points = sweeps.get(key, [])
        if not points:
            continue

        recall = [p.recall for p in points]
        precision = [p.precision for p in points]
        fpr = [p.false_positive_rate for p in points]

        scatter = ax.scatter(
            recall,
            precision,
            c=fpr,
            cmap="magma_r",
            vmin=0,
            vmax=1,
            marker=marker,
            alpha=0.75,
            label=label,
        )

    ax.set_xlabel("Recall / privacy protection")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Tradeoff Across Threshold and Weight Sweeps")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("False Positive Rate")

    savefig(fig, output_dir, "02_precision_recall_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Privacy vs false positive tradeoff
# ---------------------------------------------------------------------------

def plot_privacy_false_positive_tradeoff(
    threshold_grid: List[OperatingPoint],
    output_dir: Path,
) -> None:
    """
    The most important plot for this use case.

    X-axis:
        false positive rate among non-critical examples.

    Y-axis:
        recall/privacy protection among critical examples.

    Color:
        marked sensitive rate, showing operational burden.
    """
    if not threshold_grid:
        return

    plt = ensure_matplotlib()

    pareto = pareto_candidates(threshold_grid)

    fig, ax = plt.subplots(figsize=(10, 7))

    x = [p.false_positive_rate for p in threshold_grid]
    y = [p.recall for p in threshold_grid]
    c = [p.marked_sensitive_rate for p in threshold_grid]

    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        alpha=0.65,
        s=45,
        label="Operating points",
    )

    if pareto:
        px = [p.false_positive_rate for p in pareto]
        py = [p.recall for p in pareto]

        order = np.argsort(px)
        px = np.array(px)[order]
        py = np.array(py)[order]

        ax.plot(
            px,
            py,
            color="red",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Pareto frontier",
        )

    ax.set_xlabel("False Positive Rate on non-critical samples")
    ax.set_ylabel("Recall / privacy protection on critical samples")
    ax.set_title("Privacy vs. Over-Flagging Tradeoff")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Marked Sensitive Rate")

    savefig(fig, output_dir, "03_privacy_false_positive_tradeoff.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: k-minimum sensitivity
# ---------------------------------------------------------------------------

def plot_k_minimum_sensitivity(points: List[OperatingPoint], output_dir: Path) -> None:
    if not points:
        return

    plt = ensure_matplotlib()

    points = sorted(points, key=lambda p: p.k_min)
    arr = point_list_to_arrays(points, "k_min")

    fnr = np.array([p.false_negative_rate for p in points], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(arr["x"], arr["recall"], marker="o", label="Recall / privacy protection")
    ax.plot(arr["x"], fnr, marker="o", label="False Negative Rate")
    ax.set_ylabel("Rate")
    ax.set_title(r"Effect of $k_{\min}$ on Privacy Recall and False Negatives")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(arr["x"], arr["precision"], marker="o", label="Precision")
    ax.plot(arr["x"], arr["f1"], marker="o", label="F1")
    ax.plot(arr["x"], arr["fpr"], marker="o", label="False Positive Rate")
    ax.set_ylabel("Score / rate")
    ax.set_title(r"Effect of $k_{\min}$ on Precision and False Positives")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.stackplot(
        arr["x"],
        arr["pass"],
        arr["review"],
        arr["block"],
        labels=["Pass", "Review", "Block"],
        colors=["#2ecc71", "#f1c40f", "#e74c3c"],
        alpha=0.85,
    )
    ax.set_xlabel(r"$k_{\min}$")
    ax.set_ylabel("Routing fraction")
    ax.set_title("Routing Distribution as k-Minimum Increases")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")

    savefig(fig, output_dir, "04_k_minimum_sensitivity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Direct/quasi weight heatmaps
# ---------------------------------------------------------------------------

def plot_weight_sensitivity_heatmaps(points: List[OperatingPoint], output_dir: Path) -> None:
    if not points:
        return

    plt = ensure_matplotlib()

    w_direct_values = sorted(set(p.w_direct for p in points))
    w_quasi_values = sorted(set(p.w_quasi for p in points))

    def make_grid(metric: str) -> np.ndarray:
        grid = np.full((len(w_quasi_values), len(w_direct_values)), np.nan)
        idx_d = {v: i for i, v in enumerate(w_direct_values)}
        idx_q = {v: i for i, v in enumerate(w_quasi_values)}
        for p in points:
            grid[idx_q[p.w_quasi], idx_d[p.w_direct]] = getattr(p, metric)
        return grid

    metrics = [
        ("recall", "Recall / Privacy Protection"),
        ("false_negative_rate", "False Negative Rate"),
        ("false_positive_rate", "False Positive Rate"),
        ("precision", "Precision"),
        ("marked_sensitive_rate", "Marked Sensitive Rate"),
        ("f1", "F1"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        grid = make_grid(metric)

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[
                min(w_direct_values),
                max(w_direct_values),
                min(w_quasi_values),
                max(w_quasi_values),
            ],
            vmin=0,
            vmax=1,
            cmap="viridis",
        )

        ax.set_title(title)
        ax.set_xlabel(r"$w_{\mathrm{direct}}$")
        ax.set_ylabel(r"$w_{\mathrm{quasi}}$")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(title)

    fig.suptitle(
        "Sensitivity to Direct-Identifier and Quasi-Identifier Weights",
        fontsize=16,
    )

    savefig(fig, output_dir, "05_weight_sensitivity_heatmaps.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Operating point breakdown
# ---------------------------------------------------------------------------

def plot_operating_point_breakdown(
    chosen: Dict[str, Optional[OperatingPoint]],
    output_dir: Path,
) -> None:
    """
    Compares selected operating points.
    """
    selected = {k: v for k, v in chosen.items() if v is not None}
    if not selected:
        return

    plt = ensure_matplotlib()

    labels = list(selected.keys())
    points = list(selected.values())

    metrics = {
        "Recall": [p.recall for p in points],
        "Precision": [p.precision for p in points],
        "FPR": [p.false_positive_rate for p in points],
        "Marked sensitive": [p.marked_sensitive_rate for p in points],
    }

    x = np.arange(len(labels))
    width = 0.2

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    ax = axes[0]
    for i, (metric, values) in enumerate(metrics.items()):
        ax.bar(x + (i - 1.5) * width, values, width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score / rate")
    ax.set_ylim(0, 1)
    ax.set_title("Selected Operating Points: Metric Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    ax = axes[1]
    pass_values = [p.pass_rate for p in points]
    review_values = [p.review_rate for p in points]
    block_values = [p.block_rate for p in points]

    ax.bar(x, pass_values, label="Pass", color="#2ecc71")
    ax.bar(x, review_values, bottom=pass_values, label="Review", color="#f1c40f")
    bottom_block = np.array(pass_values) + np.array(review_values)
    ax.bar(x, block_values, bottom=bottom_block, label="Block", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Routing fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Selected Operating Points: Routing Breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    savefig(fig, output_dir, "06_operating_point_breakdown.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: Wikipedia false positive analysis
# ---------------------------------------------------------------------------

def plot_wikipedia_false_positive_analysis(
    threshold_grid: List[OperatingPoint],
    output_dir: Path,
) -> None:
    """
    Specialized plot for the user's observed problem:
    non-sensitive Wikipedia text being marked sensitive.
    """
    points = [p for p in threshold_grid if p.wiki_false_positive_rate is not None]
    if not points:
        return

    plt = ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 7))

    x = [p.recall for p in points]
    y = [p.wiki_false_positive_rate for p in points]
    c = [p.false_positive_rate for p in points]

    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="plasma",
        alpha=0.75,
        s=50,
    )

    ax.set_xlabel("Overall recall / privacy protection")
    ax.set_ylabel("Wikipedia marked-sensitive rate")
    ax.set_title("Wikipedia Over-Flagging vs. Privacy Recall")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Overall false positive rate")

    savefig(fig, output_dir, "07_wikipedia_false_positive_analysis.png")
    plt.close(fig)

def plot_false_negative_sensitivity(
    sweeps: Dict[str, List[OperatingPoint]],
    output_dir: Path,
) -> None:
    """
    Dedicated plot showing how threshold and weight changes affect
    false negative rate.

    False negatives are privacy misses:
        expected critical, but simulated route was pass.
    """
    plt = ensure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # tau_review sweep
    points = sorted(sweeps.get("tau_review_sweep", []), key=lambda p: p.tau_review)
    ax = axes[0, 0]
    if points:
        x = [p.tau_review for p in points]
        y = [p.false_negative_rate for p in points]
        recall = [p.recall for p in points]

        ax.plot(x, y, marker="o", color="#e74c3c", label="False Negative Rate")
        ax.plot(x, recall, marker="o", color="#2ecc71", label="Recall")
        ax.set_xlabel(r"$\tau_{\mathrm{review}}$")
        ax.set_ylabel("Rate")
        ax.set_title("False Negatives vs. Review Threshold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # tau_block sweep
    points = sorted(sweeps.get("tau_block_sweep", []), key=lambda p: p.tau_block)
    ax = axes[0, 1]
    if points:
        x = [p.tau_block for p in points]
        y = [p.false_negative_rate for p in points]
        recall = [p.recall for p in points]

        ax.plot(x, y, marker="o", color="#e74c3c", label="False Negative Rate")
        ax.plot(x, recall, marker="o", color="#2ecc71", label="Recall")
        ax.set_xlabel(r"$\tau_{\mathrm{block}}$")
        ax.set_ylabel("Rate")
        ax.set_title("False Negatives vs. Block Threshold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # k_min sweep
    points = sorted(sweeps.get("k_min_sweep", []), key=lambda p: p.k_min)
    ax = axes[1, 0]
    if points:
        x = [p.k_min for p in points]
        y = [p.false_negative_rate for p in points]
        recall = [p.recall for p in points]

        ax.plot(x, y, marker="o", color="#e74c3c", label="False Negative Rate")
        ax.plot(x, recall, marker="o", color="#2ecc71", label="Recall")
        ax.set_xlabel(r"$k_{\min}$")
        ax.set_ylabel("Rate")
        ax.set_title("False Negatives vs. k-Minimum")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # threshold grid: FNR/FPR tradeoff
    points = sweeps.get("threshold_grid", [])
    ax = axes[1, 1]
    if points:
        x = [p.false_positive_rate for p in points]
        y = [p.false_negative_rate for p in points]
        c = [p.marked_sensitive_rate for p in points]

        scatter = ax.scatter(
            x,
            y,
            c=c,
            cmap="viridis",
            alpha=0.75,
            s=45,
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("False Negative Rate")
        ax.set_title("False Positive vs. False Negative Tradeoff")
        ax.grid(True, alpha=0.3)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Marked Sensitive Rate")

    fig.suptitle("False Negative Sensitivity Across Thresholds", fontsize=16)

    savefig(fig, output_dir, "08_false_negative_sensitivity.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    input_path: Path,
    output_dir: Path,
    tau_review_min: float,
    tau_review_max: float,
    tau_review_step: float,
    tau_block_min: float,
    tau_block_max: float,
    tau_block_step: float,
    k_min_min: int,
    k_min_max: int,
    weight_min: float,
    weight_max: float,
    weight_step: float,
    baseline_tau_review: float,
    baseline_tau_block: float,
    baseline_k_min: int,
    baseline_w_direct: float,
    baseline_w_quasi: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(input_path)
    if not samples:
        raise ValueError("No samples found in input file.")

    tau_review_values = frange(tau_review_min, tau_review_max, tau_review_step)
    tau_block_values = frange(tau_block_min, tau_block_max, tau_block_step)
    k_min_values = list(range(k_min_min, k_min_max + 1))
    weight_values = frange(weight_min, weight_max, weight_step)

    sample_summary = summarize_samples(samples)

    baseline = evaluate_operating_point(
        samples=samples,
        tau_review=baseline_tau_review,
        tau_block=baseline_tau_block,
        k_min=baseline_k_min,
        w_direct=baseline_w_direct,
        w_quasi=baseline_w_quasi,
    )

    sweeps = run_sweeps(
        samples=samples,
        tau_review_values=tau_review_values,
        tau_block_values=tau_block_values,
        k_min_values=k_min_values,
        weight_values=weight_values,
        baseline_tau_review=baseline_tau_review,
        baseline_tau_block=baseline_tau_block,
        baseline_k_min=baseline_k_min,
        baseline_w_direct=baseline_w_direct,
        baseline_w_quasi=baseline_w_quasi,
    )

    all_points = []
    for points in sweeps.values():
        all_points.extend(points)

    chosen = choose_operating_points(all_points)

    # Write machine-readable outputs.
    sweep_json = {
        name: [asdict(p) for p in points]
        for name, points in sweeps.items()
    }

    write_json(output_dir / "threshold_sweep_results.json", sweep_json)

    summary_json = {
        "sample_summary": sample_summary,
        "baseline": asdict(baseline),
        "chosen_operating_points": {
            name: asdict(point) if point is not None else None
            for name, point in chosen.items()
        },
        "sweep_config": {
            "tau_review_values": tau_review_values,
            "tau_block_values": tau_block_values,
            "k_min_values": k_min_values,
            "weight_values": weight_values,
            "baseline_tau_review": baseline_tau_review,
            "baseline_tau_block": baseline_tau_block,
            "baseline_k_min": baseline_k_min,
            "baseline_w_direct": baseline_w_direct,
            "baseline_w_quasi": baseline_w_quasi,
        },
    }

    write_json(output_dir / "analysis_summary.json", summary_json)

    # Plots.
    plot_threshold_phase_diagram(sweeps["threshold_grid"], output_dir)
    plot_precision_recall_curves(sweeps, output_dir)
    plot_privacy_false_positive_tradeoff(sweeps["threshold_grid"], output_dir)
    plot_k_minimum_sensitivity(sweeps["k_min_sweep"], output_dir)
    plot_weight_sensitivity_heatmaps(sweeps["weight_grid"], output_dir)
    plot_operating_point_breakdown(chosen, output_dir)
    plot_wikipedia_false_positive_analysis(sweeps["threshold_grid"], output_dir)
    plot_false_negative_sensitivity(sweeps, output_dir)

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print()
    print("Baseline:")
    print(f"  tau_review: {baseline.tau_review:.2f}")
    print(f"  tau_block:  {baseline.tau_block:.2f}")
    print(f"  k_min:      {baseline.k_min}")
    print(f"  w_direct:   {baseline.w_direct:.2f}")
    print(f"  w_quasi:    {baseline.w_quasi:.2f}")
    print(f"  precision:  {baseline.precision:.3f}")
    print(f"  recall:     {baseline.recall:.3f}")
    print(f"  f1:         {baseline.f1:.3f}")
    print(f"  fpr:        {baseline.false_positive_rate:.3f}")
    print(f"  marked:     {baseline.marked_sensitive_rate:.3f}")
    print()

    for name, point in chosen.items():
        if point is None:
            continue
        print(f"{name}:")
        print(f"  tau_review={point.tau_review:.2f}, tau_block={point.tau_block:.2f}, "
              f"k_min={point.k_min}, w_direct={point.w_direct:.2f}, "
              f"w_quasi={point.w_quasi:.2f}")
        print(f"  precision={point.precision:.3f}, recall={point.recall:.3f}, "
              f"f1={point.f1:.3f}, fpr={point.false_positive_rate:.3f}, "
              f"marked={point.marked_sensitive_rate:.3f}")
        if point.wiki_false_positive_rate is not None:
            print(f"  wiki_marked_sensitive_rate={point.wiki_false_positive_rate:.3f}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze threshold sensitivity for privacy routing benchmark outputs."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("pipeline_results.json"),
        help="Path to pipeline_results.json.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory for plots and analysis outputs.",
    )

    # Requested threshold ranges.
    parser.add_argument("--tau-review-min", type=float, default=0.01)
    parser.add_argument("--tau-review-max", type=float, default=0.50)
    parser.add_argument("--tau-review-step", type=float, default=0.02)

    parser.add_argument("--tau-block-min", type=float, default=0.50)
    parser.add_argument("--tau-block-max", type=float, default=0.95)
    parser.add_argument("--tau-block-step", type=float, default=0.05)

    parser.add_argument("--k-min-min", type=int, default=2)
    parser.add_argument("--k-min-max", type=int, default=20)

    parser.add_argument("--weight-min", type=float, default=0.1)
    parser.add_argument("--weight-max", type=float, default=1.0)
    parser.add_argument("--weight-step", type=float, default=0.1)

    # Baseline values used when a parameter family is not being swept.
    parser.add_argument("--baseline-tau-review", type=float, default=0.30)
    parser.add_argument("--baseline-tau-block", type=float, default=0.70)
    parser.add_argument("--baseline-k-min", type=int, default=5)
    parser.add_argument("--baseline-w-direct", type=float, default=0.50)
    parser.add_argument("--baseline-w-quasi", type=float, default=0.50)

    args = parser.parse_args()

    run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        tau_review_min=args.tau_review_min,
        tau_review_max=args.tau_review_max,
        tau_review_step=args.tau_review_step,
        tau_block_min=args.tau_block_min,
        tau_block_max=args.tau_block_max,
        tau_block_step=args.tau_block_step,
        k_min_min=args.k_min_min,
        k_min_max=args.k_min_max,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        weight_step=args.weight_step,
        baseline_tau_review=args.baseline_tau_review,
        baseline_tau_block=args.baseline_tau_block,
        baseline_k_min=args.baseline_k_min,
        baseline_w_direct=args.baseline_w_direct,
        baseline_w_quasi=args.baseline_w_quasi,
    )


if __name__ == "__main__":
    main()