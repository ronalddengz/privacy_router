#!/usr/bin/env python3
"""
analysis.py - Threshold and weight sweep analysis for privacy router benchmark results.

Reads saved pipeline_results.json and simulates alternate routing decisions
under different threshold/weight configurations to find optimal operating points.

Usage:
    python analysis.py -i pipeline_results.json -o analysis_outputs
    python analysis.py -i benchmark_revised_outputs/pipeline_results.json -o analysis_outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineSample:
    """Loaded pipeline result for sweep analysis."""
    sample_name: str
    sample_hash: str
    text_length: int
    expected_critical: bool
    tier: int
    gate_decision: str
    k_lower: float
    k_upper: float
    joint_risk_score: float
    direct_identifier_count: int
    quasi_identifier_count: int
    llm_invoked: bool
    masking_ratio: float
    word_count: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineSample":
        word_count = d.get("word_count", 0)
        if word_count == 0:
            # Estimate from text length (~5 chars per word)
            word_count = max(1, d.get("text_length", 100) // 5)
        return cls(
            sample_name=d.get("sample_name", ""),
            sample_hash=d.get("sample_hash", ""),
            text_length=d.get("text_length", 0),
            expected_critical=bool(d.get("expected_critical", False)),
            tier=d.get("tier", 0),
            gate_decision=d.get("gate_decision", ""),
            k_lower=float(d.get("k_lower", 0.0)),
            k_upper=float(d.get("k_upper", float("inf"))),
            joint_risk_score=float(d.get("joint_risk_score", 0.0)),
            direct_identifier_count=int(d.get("direct_identifier_count", 0)),
            quasi_identifier_count=int(d.get("quasi_identifier_count", 0)),
            llm_invoked=bool(d.get("llm_invoked", False)),
            masking_ratio=float(d.get("masking_ratio", 0.0)),
            word_count=word_count,
        )

    def is_wikipedia(self) -> bool:
        return "wiki" in self.sample_name.lower()


@dataclass
class OperatingPoint:
    """Metrics for a single operating point configuration."""
    sweep_name: str = ""

    # Threshold parameters
    tau_review: float = 0.30
    tau_block: float = 0.70
    k_min: int = 5
    k_safe_threshold: int = 20

    # Separate direct/quasi thresholds
    tau_direct_review: float = 0.20
    tau_direct_block: float = 0.60
    tau_quasi_review: float = 0.30
    tau_quasi_block: float = 0.70

    # Weight parameters
    w_direct: float = 0.50
    w_quasi: float = 0.50

    # Classification metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0

    # Confusion matrix
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    # Routing rates
    marked_sensitive_rate: float = 0.0
    pass_rate: float = 0.0
    review_rate: float = 0.0
    block_rate: float = 0.0

    # Wikipedia-specific
    wiki_false_positive_rate: float = 0.0
    wiki_marked_sensitive_count: int = 0
    wiki_total: int = 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom > 0 else default


def load_pipeline_results(path: Path) -> List[PipelineSample]:
    with open(path, "r") as f:
        data = json.load(f)
    return [PipelineSample.from_dict(d) for d in data]


def write_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not available, skipping plots", file=sys.stderr)
        return None


def savefig(fig, output_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    print(f"  Saved {filename}")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_normalization(samples: List[PipelineSample]) -> Tuple[float, float]:
    """Compute max counts for normalization."""
    max_direct = max((s.direct_identifier_count for s in samples), default=1)
    max_quasi = max((s.quasi_identifier_count for s in samples), default=1)
    return float(max(max_direct, 1)), float(max(max_quasi, 1))


# ---------------------------------------------------------------------------
# Routing simulation with separate direct/quasi thresholds
# ---------------------------------------------------------------------------

def simulate_routing(
    sample: PipelineSample,
    max_direct: float,
    max_quasi: float,
    # Global thresholds (used for combined risk)
    tau_review: float,
    tau_block: float,
    k_min: int,
    k_safe_threshold: int,
    # Separate direct/quasi thresholds
    tau_direct_review: float,
    tau_direct_block: float,
    tau_quasi_review: float,
    tau_quasi_block: float,
    # Weights for combined score
    w_direct: float,
    w_quasi: float,
) -> Tuple[bool, bool, bool]:
    """
    Simulate routing decision with separate direct/quasi thresholds.

    Returns: (marked_sensitive, is_review, is_block)
    """
    # Normalize scores
    direct_score = sample.direct_identifier_count / max_direct
    quasi_score = sample.quasi_identifier_count / max_quasi

    # Combined risk score (weighted)
    combined_risk = (
        sample.joint_risk_score +
        w_direct * direct_score +
        w_quasi * quasi_score
    ) / (1.0 + w_direct + w_quasi)

    # === Decision logic ===

    # Hard block conditions
    # 1. k_lower below minimum
    if sample.k_lower < k_min:
        return True, False, True

    # 2. Direct identifier score above direct block threshold
    if direct_score >= tau_direct_block:
        return True, False, True

    # 3. Quasi identifier score above quasi block threshold
    if quasi_score >= tau_quasi_block:
        return True, False, True

    # 4. Combined risk above global block threshold
    if combined_risk >= tau_block:
        return True, False, True

    # Review conditions
    # 1. Direct score above direct review threshold
    if direct_score >= tau_direct_review:
        return True, True, False

    # 2. Quasi score above quasi review threshold
    if quasi_score >= tau_quasi_review:
        return True, True, False

    # 3. Combined risk above global review threshold
    if combined_risk >= tau_review:
        return True, True, False

    # 4. k_lower below safe threshold but above minimum -> review
    if sample.k_lower < k_safe_threshold:
        return True, True, False

    # Safe to pass
    return False, False, False


def evaluate_operating_point(
    samples: List[PipelineSample],
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
    """Evaluate metrics for a single operating point."""
    tp = fp = tn = fn = 0
    review_count = 0
    block_count = 0
    wiki_marked = 0
    wiki_total = 0

    for s in samples:
        marked, is_review, is_block = simulate_routing(
            sample=s,
            max_direct=max_direct,
            max_quasi=max_quasi,
            tau_review=tau_review,
            tau_block=tau_block,
            k_min=k_min,
            k_safe_threshold=k_safe_threshold,
            tau_direct_review=tau_direct_review,
            tau_direct_block=tau_direct_block,
            tau_quasi_review=tau_quasi_review,
            tau_quasi_block=tau_quasi_block,
            w_direct=w_direct,
            w_quasi=w_quasi,
        )

        actual = s.expected_critical

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

        if s.is_wikipedia():
            wiki_total += 1
            if marked:
                wiki_marked += 1

    total = len(samples)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    accuracy = safe_div(tp + tn, total)
    balanced_accuracy = (recall + specificity) / 2

    return OperatingPoint(
        sweep_name=sweep_name,
        tau_review=tau_review,
        tau_block=tau_block,
        k_min=k_min,
        k_safe_threshold=k_safe_threshold,
        tau_direct_review=tau_direct_review,
        tau_direct_block=tau_direct_block,
        tau_quasi_review=tau_quasi_review,
        tau_quasi_block=tau_quasi_block,
        w_direct=w_direct,
        w_quasi=w_quasi,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        marked_sensitive_rate=safe_div(tp + fp, total),
        pass_rate=safe_div(tn + fn, total),
        review_rate=safe_div(review_count, total),
        block_rate=safe_div(block_count, total),
        wiki_false_positive_rate=safe_div(wiki_marked, wiki_total),
        wiki_marked_sensitive_count=wiki_marked,
        wiki_total=wiki_total,
    )


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------

def run_sweeps(
    samples: List[PipelineSample],
    # Global threshold ranges
    tau_review_values: List[float],
    tau_block_values: List[float],
    k_min_values: List[int],
    k_safe_threshold_values: List[int],
    # Separate direct/quasi threshold ranges
    tau_direct_review_values: List[float],
    tau_direct_block_values: List[float],
    tau_quasi_review_values: List[float],
    tau_quasi_block_values: List[float],
    # Weight ranges
    weight_values: List[float],
    # Baselines
    baseline_tau_review: float,
    baseline_tau_block: float,
    baseline_k_min: int,
    baseline_k_safe_threshold: int,
    baseline_tau_direct_review: float,
    baseline_tau_direct_block: float,
    baseline_tau_quasi_review: float,
    baseline_tau_quasi_block: float,
    baseline_w_direct: float,
    baseline_w_quasi: float,
) -> Dict[str, List[OperatingPoint]]:
    """Run all configured sweeps."""
    max_direct, max_quasi = compute_normalization(samples)
    results: Dict[str, List[OperatingPoint]] = {}

    def eval_point(name: str, **kwargs) -> OperatingPoint:
        defaults = dict(
            tau_review=baseline_tau_review,
            tau_block=baseline_tau_block,
            k_min=baseline_k_min,
            k_safe_threshold=baseline_k_safe_threshold,
            tau_direct_review=baseline_tau_direct_review,
            tau_direct_block=baseline_tau_direct_block,
            tau_quasi_review=baseline_tau_quasi_review,
            tau_quasi_block=baseline_tau_quasi_block,
            w_direct=baseline_w_direct,
            w_quasi=baseline_w_quasi,
        )
        defaults.update(kwargs)
        return evaluate_operating_point(
            samples=samples,
            max_direct=max_direct,
            max_quasi=max_quasi,
            sweep_name=name,
            **defaults,
        )

    # 1. tau_review sweep
    print("  Running tau_review sweep...")
    results["tau_review_sweep"] = [
        eval_point("tau_review_sweep", tau_review=v)
        for v in tau_review_values
    ]

    # 2. tau_block sweep
    print("  Running tau_block sweep...")
    results["tau_block_sweep"] = [
        eval_point("tau_block_sweep", tau_block=v)
        for v in tau_block_values
    ]

    # 3. k_min sweep
    print("  Running k_min sweep...")
    results["k_min_sweep"] = [
        eval_point("k_min_sweep", k_min=v)
        for v in k_min_values
    ]

    # 4. k_safe_threshold sweep (NEW)
    print("  Running k_safe_threshold sweep...")
    results["k_safe_threshold_sweep"] = [
        eval_point("k_safe_threshold_sweep", k_safe_threshold=v)
        for v in k_safe_threshold_values
    ]

    # 5. tau_direct_review sweep (NEW)
    print("  Running tau_direct_review sweep...")
    results["tau_direct_review_sweep"] = [
        eval_point("tau_direct_review_sweep", tau_direct_review=v)
        for v in tau_direct_review_values
    ]

    # 6. tau_direct_block sweep (NEW)
    print("  Running tau_direct_block sweep...")
    results["tau_direct_block_sweep"] = [
        eval_point("tau_direct_block_sweep", tau_direct_block=v)
        for v in tau_direct_block_values
    ]

    # 7. tau_quasi_review sweep (NEW)
    print("  Running tau_quasi_review sweep...")
    results["tau_quasi_review_sweep"] = [
        eval_point("tau_quasi_review_sweep", tau_quasi_review=v)
        for v in tau_quasi_review_values
    ]

    # 8. tau_quasi_block sweep (NEW)
    print("  Running tau_quasi_block sweep...")
    results["tau_quasi_block_sweep"] = [
        eval_point("tau_quasi_block_sweep", tau_quasi_block=v)
        for v in tau_quasi_block_values
    ]

    # 9. Weight grid (w_direct x w_quasi)
    print("  Running weight grid sweep...")
    weight_grid = []
    for wd in weight_values:
        for wq in weight_values:
            weight_grid.append(eval_point("weight_grid", w_direct=wd, w_quasi=wq))
    results["weight_grid"] = weight_grid

    # 10. Direct threshold grid (tau_direct_review x tau_direct_block) (NEW)
    print("  Running direct threshold grid sweep...")
    direct_grid = []
    for tr in tau_direct_review_values[::2]:  # Subsample for speed
        for tb in tau_direct_block_values[::2]:
            if tr < tb:
                direct_grid.append(eval_point(
                    "direct_threshold_grid",
                    tau_direct_review=tr,
                    tau_direct_block=tb,
                ))
    results["direct_threshold_grid"] = direct_grid

    # 11. Quasi threshold grid (tau_quasi_review x tau_quasi_block) (NEW)
    print("  Running quasi threshold grid sweep...")
    quasi_grid = []
    for tr in tau_quasi_review_values[::2]:  # Subsample for speed
        for tb in tau_quasi_block_values[::2]:
            if tr < tb:
                quasi_grid.append(eval_point(
                    "quasi_threshold_grid",
                    tau_quasi_review=tr,
                    tau_quasi_block=tb,
                ))
    results["quasi_threshold_grid"] = quasi_grid

    # 12. Full threshold grid (global tau_review x tau_block x k_safe)
    print("  Running full threshold grid sweep...")
    threshold_grid = []
    for tr in tau_review_values[::3]:
        for tb in tau_block_values[::2]:
            for ks in k_safe_threshold_values[::2]:
                if tr < tb:
                    threshold_grid.append(eval_point(
                        "threshold_grid",
                        tau_review=tr,
                        tau_block=tb,
                        k_safe_threshold=ks,
                    ))
    results["threshold_grid"] = threshold_grid

    # 13. Combined grid: direct/quasi thresholds + k_safe (NEW)
    print("  Running combined direct/quasi/k_safe grid sweep...")
    combined_grid = []
    for tdr in [0.10, 0.20, 0.30, 0.40]:
        for tdb in [0.50, 0.60, 0.70, 0.80]:
            for tqr in [0.20, 0.30, 0.40]:
                for tqb in [0.60, 0.70, 0.80]:
                    for ks in [10, 20, 50, 100]:
                        if tdr < tdb and tqr < tqb:
                            combined_grid.append(eval_point(
                                "combined_grid",
                                tau_direct_review=tdr,
                                tau_direct_block=tdb,
                                tau_quasi_review=tqr,
                                tau_quasi_block=tqb,
                                k_safe_threshold=ks,
                            ))
    results["combined_grid"] = combined_grid

    return results


# ---------------------------------------------------------------------------
# Operating point selection
# ---------------------------------------------------------------------------

def choose_operating_points(
    points: List[OperatingPoint],
) -> Dict[str, OperatingPoint]:
    """Select notable operating points."""
    if not points:
        return {}

    chosen = {}

    # Best F1
    best_f1 = max(points, key=lambda p: p.f1)
    chosen["best_f1"] = best_f1

    # Best balanced accuracy
    best_ba = max(points, key=lambda p: p.balanced_accuracy)
    chosen["best_balanced_accuracy"] = best_ba

    # Highest recall with FPR <= 0.15
    valid = [p for p in points if p.false_positive_rate <= 0.15]
    if valid:
        chosen["high_recall_fpr_le_15"] = max(valid, key=lambda p: p.recall)

    # Highest recall with FPR <= 0.10
    valid = [p for p in points if p.false_positive_rate <= 0.10]
    if valid:
        chosen["high_recall_fpr_le_10"] = max(valid, key=lambda p: p.recall)

    # Highest recall with wiki FPR <= 0.10
    valid = [p for p in points if p.wiki_false_positive_rate <= 0.10]
    if valid:
        chosen["high_recall_wiki_fpr_le_10"] = max(valid, key=lambda p: p.recall)

    # Lowest FNR with FPR <= 0.15
    valid = [p for p in points if p.false_positive_rate <= 0.15]
    if valid:
        chosen["low_fnr_fpr_le_15"] = min(valid, key=lambda p: p.false_negative_rate)

    # Lowest FNR with wiki FPR <= 0.10
    valid = [p for p in points if p.wiki_false_positive_rate <= 0.10]
    if valid:
        chosen["low_fnr_wiki_fpr_le_10"] = min(valid, key=lambda p: p.false_negative_rate)

    # Lowest wiki FPR with recall >= 0.90
    valid = [p for p in points if p.recall >= 0.90]
    if valid:
        chosen["low_wiki_fpr_recall_ge_90"] = min(valid, key=lambda p: p.wiki_false_positive_rate)

    # Lowest wiki FPR with recall >= 0.95
    valid = [p for p in points if p.recall >= 0.95]
    if valid:
        chosen["low_wiki_fpr_recall_ge_95"] = min(valid, key=lambda p: p.wiki_false_positive_rate)

    # Lowest wiki FPR overall
    chosen["lowest_wiki_fpr"] = min(points, key=lambda p: p.wiki_false_positive_rate)

    return chosen


def pareto_candidates(points: List[OperatingPoint]) -> List[OperatingPoint]:
    """Find Pareto-optimal points minimizing FPR and FNR."""
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if (q.false_positive_rate <= p.false_positive_rate and
                q.false_negative_rate <= p.false_negative_rate and
                (q.false_positive_rate < p.false_positive_rate or
                 q.false_negative_rate < p.false_negative_rate)):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return pareto


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_1d_sweep(
    points: List[OperatingPoint],
    x_attr: str,
    x_label: str,
    title: str,
    output_dir: Path,
    filename: str,
) -> None:
    """Plot a 1D sweep showing key metrics vs parameter."""
    plt = ensure_matplotlib()
    if plt is None or not points:
        return

    points = sorted(points, key=lambda p: getattr(p, x_attr))
    x = [getattr(p, x_attr) for p in points]
    recall = [p.recall for p in points]
    fnr = [p.false_negative_rate for p in points]
    fpr = [p.false_positive_rate for p in points]
    wiki_fpr = [p.wiki_false_positive_rate for p in points]
    f1 = [p.f1 for p in points]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(x, recall, "g-o", label="Recall (privacy protection)", linewidth=2)
    ax.plot(x, fnr, "r-s", label="False Negative Rate", linewidth=2)
    ax.plot(x, f1, "b-^", label="F1", linewidth=2, alpha=0.7)
    ax.set_ylabel("Rate / Score")
    ax.set_title(f"{title} - Recall and False Negatives")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, fpr, "m-o", label="False Positive Rate", linewidth=2)
    ax.plot(x, wiki_fpr, "c-s", label="Wikipedia FPR", linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rate")
    ax.set_title(f"{title} - False Positives")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    savefig(fig, output_dir, filename)
    plt.close(fig)


def plot_threshold_heatmap(
    points: List[OperatingPoint],
    x_attr: str,
    y_attr: str,
    x_label: str,
    y_label: str,
    title: str,
    output_dir: Path,
    filename: str,
) -> None:
    """Plot heatmaps for a 2D threshold grid."""
    plt = ensure_matplotlib()
    if plt is None or not points:
        return

    x_vals = sorted(set(getattr(p, x_attr) for p in points))
    y_vals = sorted(set(getattr(p, y_attr) for p in points))

    if len(x_vals) < 2 or len(y_vals) < 2:
        return

    # Create lookup
    lookup = {}
    for p in points:
        key = (getattr(p, x_attr), getattr(p, y_attr))
        lookup[key] = p

    metrics = ["recall", "false_negative_rate", "false_positive_rate", "wiki_false_positive_rate", "f1"]
    metric_labels = ["Recall", "False Negative Rate", "False Positive Rate", "Wiki FPR", "F1"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if idx >= len(axes):
            break
        ax = axes[idx]

        data = np.zeros((len(y_vals), len(x_vals)))
        for i, yv in enumerate(y_vals):
            for j, xv in enumerate(x_vals):
                p = lookup.get((xv, yv))
                data[i, j] = getattr(p, metric) if p else np.nan

        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[min(x_vals), max(x_vals), min(y_vals), max(y_vals)],
                       cmap="RdYlGn" if metric in ["recall", "f1"] else "RdYlGn_r")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(label)
        fig.colorbar(im, ax=ax)

    # Remove empty subplot
    if len(metrics) < len(axes):
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    savefig(fig, output_dir, filename)
    plt.close(fig)


def plot_pareto_frontier(
    points: List[OperatingPoint],
    output_dir: Path,
) -> None:
    """Plot FPR vs FNR with Pareto frontier."""
    plt = ensure_matplotlib()
    if plt is None or not points:
        return

    pareto = pareto_candidates(points)

    fig, ax = plt.subplots(figsize=(10, 8))

    fpr = [p.false_positive_rate for p in points]
    fnr = [p.false_negative_rate for p in points]
    recall = [p.recall for p in points]

    scatter = ax.scatter(fpr, fnr, c=recall, cmap="viridis", alpha=0.5, s=30)

    if pareto:
        pareto = sorted(pareto, key=lambda p: p.false_positive_rate)
        pfpr = [p.false_positive_rate for p in pareto]
        pfnr = [p.false_negative_rate for p in pareto]
        ax.plot(pfpr, pfnr, "r-o", linewidth=2, markersize=8, label="Pareto frontier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("FPR vs FNR Tradeoff with Pareto Frontier")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Recall")

    savefig(fig, output_dir, "pareto_frontier.png")
    plt.close(fig)


def plot_direct_quasi_comparison(
    sweeps: Dict[str, List[OperatingPoint]],
    output_dir: Path,
) -> None:
    """Compare direct vs quasi threshold effects."""
    plt = ensure_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Direct review sweep
    ax = axes[0, 0]
    points = sweeps.get("tau_direct_review_sweep", [])
    if points:
        points = sorted(points, key=lambda p: p.tau_direct_review)
        x = [p.tau_direct_review for p in points]
        ax.plot(x, [p.recall for p in points], "g-o", label="Recall")
        ax.plot(x, [p.false_negative_rate for p in points], "r-s", label="FNR")
        ax.plot(x, [p.wiki_false_positive_rate for p in points], "c-^", label="Wiki FPR")
        ax.set_xlabel("tau_direct_review")
        ax.set_ylabel("Rate")
        ax.set_title("Direct Identifier Review Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Direct block sweep
    ax = axes[0, 1]
    points = sweeps.get("tau_direct_block_sweep", [])
    if points:
        points = sorted(points, key=lambda p: p.tau_direct_block)
        x = [p.tau_direct_block for p in points]
        ax.plot(x, [p.recall for p in points], "g-o", label="Recall")
        ax.plot(x, [p.false_negative_rate for p in points], "r-s", label="FNR")
        ax.plot(x, [p.wiki_false_positive_rate for p in points], "c-^", label="Wiki FPR")
        ax.set_xlabel("tau_direct_block")
        ax.set_ylabel("Rate")
        ax.set_title("Direct Identifier Block Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Quasi review sweep
    ax = axes[1, 0]
    points = sweeps.get("tau_quasi_review_sweep", [])
    if points:
        points = sorted(points, key=lambda p: p.tau_quasi_review)
        x = [p.tau_quasi_review for p in points]
        ax.plot(x, [p.recall for p in points], "g-o", label="Recall")
        ax.plot(x, [p.false_negative_rate for p in points], "r-s", label="FNR")
        ax.plot(x, [p.wiki_false_positive_rate for p in points], "c-^", label="Wiki FPR")
        ax.set_xlabel("tau_quasi_review")
        ax.set_ylabel("Rate")
        ax.set_title("Quasi Identifier Review Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Quasi block sweep
    ax = axes[1, 1]
    points = sweeps.get("tau_quasi_block_sweep", [])
    if points:
        points = sorted(points, key=lambda p: p.tau_quasi_block)
        x = [p.tau_quasi_block for p in points]
        ax.plot(x, [p.recall for p in points], "g-o", label="Recall")
        ax.plot(x, [p.false_negative_rate for p in points], "r-s", label="FNR")
        ax.plot(x, [p.wiki_false_positive_rate for p in points], "c-^", label="Wiki FPR")
        ax.set_xlabel("tau_quasi_block")
        ax.set_ylabel("Rate")
        ax.set_title("Quasi Identifier Block Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Direct vs Quasi Identifier Threshold Effects", fontsize=14, fontweight="bold")
    savefig(fig, output_dir, "direct_quasi_threshold_comparison.png")
    plt.close(fig)


def plot_k_safe_threshold_sweep(
    points: List[OperatingPoint],
    output_dir: Path,
) -> None:
    """Plot k_safe_threshold sweep results."""
    plt = ensure_matplotlib()
    if plt is None or not points:
        return

    points = sorted(points, key=lambda p: p.k_safe_threshold)
    x = [p.k_safe_threshold for p in points]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(x, [p.recall for p in points], "g-o", label="Recall", linewidth=2)
    ax.plot(x, [p.false_negative_rate for p in points], "r-s", label="FNR", linewidth=2)
    ax.plot(x, [p.f1 for p in points], "b-^", label="F1", linewidth=2, alpha=0.7)
    ax.set_ylabel("Rate / Score")
    ax.set_title("k_safe_threshold Effect on Privacy Protection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, [p.false_positive_rate for p in points], "m-o", label="FPR", linewidth=2)
    ax.plot(x, [p.wiki_false_positive_rate for p in points], "c-s", label="Wiki FPR", linewidth=2)
    ax.plot(x, [p.review_rate for p in points], "y-^", label="Review Rate", linewidth=2, alpha=0.7)
    ax.set_xlabel("k_safe_threshold")
    ax.set_ylabel("Rate")
    ax.set_title("k_safe_threshold Effect on False Positives")
    ax.legend()
    ax.grid(True, alpha=0.3)

    savefig(fig, output_dir, "k_safe_threshold_sweep.png")
    plt.close(fig)


def create_all_plots(
    sweeps: Dict[str, List[OperatingPoint]],
    output_dir: Path,
) -> None:
    """Create all analysis plots."""
    print("Creating plots...")

    # 1D sweeps
    if "tau_review_sweep" in sweeps:
        plot_1d_sweep(
            sweeps["tau_review_sweep"],
            "tau_review", "tau_review",
            "Global Review Threshold",
            output_dir, "01_tau_review_sweep.png"
        )

    if "tau_block_sweep" in sweeps:
        plot_1d_sweep(
            sweeps["tau_block_sweep"],
            "tau_block", "tau_block",
            "Global Block Threshold",
            output_dir, "02_tau_block_sweep.png"
        )

    if "k_min_sweep" in sweeps:
        plot_1d_sweep(
            sweeps["k_min_sweep"],
            "k_min", "k_min",
            "Minimum k-Anonymity",
            output_dir, "03_k_min_sweep.png"
        )

    if "k_safe_threshold_sweep" in sweeps:
        plot_k_safe_threshold_sweep(
            sweeps["k_safe_threshold_sweep"],
            output_dir
        )

    # Direct/quasi comparison
    plot_direct_quasi_comparison(sweeps, output_dir)

    # 2D heatmaps
    if "direct_threshold_grid" in sweeps:
        plot_threshold_heatmap(
            sweeps["direct_threshold_grid"],
            "tau_direct_review", "tau_direct_block",
            "tau_direct_review", "tau_direct_block",
            "Direct Identifier Threshold Grid",
            output_dir, "05_direct_threshold_heatmap.png"
        )

    if "quasi_threshold_grid" in sweeps:
        plot_threshold_heatmap(
            sweeps["quasi_threshold_grid"],
            "tau_quasi_review", "tau_quasi_block",
            "tau_quasi_review", "tau_quasi_block",
            "Quasi Identifier Threshold Grid",
            output_dir, "06_quasi_threshold_heatmap.png"
        )

    if "weight_grid" in sweeps:
        plot_threshold_heatmap(
            sweeps["weight_grid"],
            "w_direct", "w_quasi",
            "w_direct", "w_quasi",
            "Weight Grid",
            output_dir, "07_weight_heatmap.png"
        )

    # Pareto frontier from combined grid
    all_points = []
    for pts in sweeps.values():
        all_points.extend(pts)
    if all_points:
        plot_pareto_frontier(all_points, output_dir)


# ---------------------------------------------------------------------------
# Summary and LaTeX generation
# ---------------------------------------------------------------------------

def generate_summary(
    samples: List[PipelineSample],
    sweeps: Dict[str, List[OperatingPoint]],
    baseline: OperatingPoint,
    chosen: Dict[str, OperatingPoint],
) -> Dict[str, Any]:
    """Generate analysis summary."""
    wiki_samples = [s for s in samples if s.is_wikipedia()]

    return {
        "sample_summary": {
            "total_samples": len(samples),
            "expected_critical": sum(1 for s in samples if s.expected_critical),
            "expected_noncritical": sum(1 for s in samples if not s.expected_critical),
            "wikipedia_samples": len(wiki_samples),
            "mean_direct_identifier_count": float(np.mean([s.direct_identifier_count for s in samples])),
            "mean_quasi_identifier_count": float(np.mean([s.quasi_identifier_count for s in samples])),
            "mean_k_lower": float(np.mean([s.k_lower for s in samples])),
        },
        "baseline": asdict(baseline),
        "chosen_operating_points": {k: asdict(v) for k, v in chosen.items()},
        "sweep_sizes": {k: len(v) for k, v in sweeps.items()},
    }


def generate_latex_report(
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate LaTeX report."""
    chosen = summary.get("chosen_operating_points", {})

    latex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}

\title{Privacy Router Threshold Analysis\\with Separate Direct/Quasi Thresholds}
\author{Automated Analysis}
\date{\today}

\begin{document}
\maketitle

\section{Overview}

This analysis evaluates the privacy router under different threshold configurations,
including \textbf{separate thresholds for direct identifiers vs quasi-identifiers}
and a \textbf{k\_safe\_threshold} parameter.

\subsection{New Parameters}

\begin{itemize}
    \item \texttt{tau\_direct\_review}: Review threshold for direct identifier score
    \item \texttt{tau\_direct\_block}: Block threshold for direct identifier score
    \item \texttt{tau\_quasi\_review}: Review threshold for quasi identifier score
    \item \texttt{tau\_quasi\_block}: Block threshold for quasi identifier score
    \item \texttt{k\_safe\_threshold}: k-anonymity level below which review is triggered (even if above k\_min)
\end{itemize}

\subsection{Routing Logic}

The routing decision follows this priority:
\begin{enumerate}
    \item \textbf{Block} if $k_{\text{lower}} < k_{\min}$
    \item \textbf{Block} if direct score $\geq \tau_{\text{direct\_block}}$
    \item \textbf{Block} if quasi score $\geq \tau_{\text{quasi\_block}}$
    \item \textbf{Block} if combined risk $\geq \tau_{\text{block}}$
    \item \textbf{Review} if direct score $\geq \tau_{\text{direct\_review}}$
    \item \textbf{Review} if quasi score $\geq \tau_{\text{quasi\_review}}$
    \item \textbf{Review} if combined risk $\geq \tau_{\text{review}}$
    \item \textbf{Review} if $k_{\text{lower}} < k_{\text{safe\_threshold}}$
    \item \textbf{Pass} otherwise
\end{enumerate}

\section{Dataset Summary}

"""
    ss = summary.get("sample_summary", {})
    latex += f"""
\\begin{{itemize}}
    \\item Total samples: {ss.get('total_samples', 0)}
    \\item Expected critical: {ss.get('expected_critical', 0)}
    \\item Expected non-critical: {ss.get('expected_noncritical', 0)}
    \\item Wikipedia samples: {ss.get('wikipedia_samples', 0)}
    \\item Mean direct identifier count: {ss.get('mean_direct_identifier_count', 0):.2f}
    \\item Mean quasi identifier count: {ss.get('mean_quasi_identifier_count', 0):.2f}
    \\item Mean k\_lower: {ss.get('mean_k_lower', 0):.2f}
\\end{{itemize}}

\\section{{Selected Operating Points}}

"""

    for name, point in chosen.items():
        latex += f"""
\\subsection{{{name.replace('_', ' ').title()}}}

\\begin{{tabular}}{{ll}}
\\toprule
Parameter & Value \\\\
\\midrule
tau\_direct\_review & {point.get('tau_direct_review', 0):.2f} \\\\
tau\_direct\_block & {point.get('tau_direct_block', 0):.2f} \\\\
tau\_quasi\_review & {point.get('tau_quasi_review', 0):.2f} \\\\
tau\_quasi\_block & {point.get('tau_quasi_block', 0):.2f} \\\\
k\_safe\_threshold & {point.get('k_safe_threshold', 0)} \\\\
w\_direct & {point.get('w_direct', 0):.2f} \\\\
w\_quasi & {point.get('w_quasi', 0):.2f} \\\\
\\midrule
Recall & {point.get('recall', 0):.4f} \\\\
False Negative Rate & {point.get('false_negative_rate', 0):.4f} \\\\
False Positive Rate & {point.get('false_positive_rate', 0):.4f} \\\\
Wiki FPR & {point.get('wiki_false_positive_rate', 0):.4f} \\\\
F1 & {point.get('f1', 0):.4f} \\\\
\\bottomrule
\\end{{tabular}}

"""

    latex += r"""
\section{Plots}

\subsection{Direct vs Quasi Threshold Comparison}
\includegraphics[width=\textwidth]{direct_quasi_threshold_comparison.png}

\subsection{k\_safe\_threshold Sweep}
\includegraphics[width=\textwidth]{k_safe_threshold_sweep.png}

\subsection{Pareto Frontier}
\includegraphics[width=\textwidth]{pareto_frontier.png}

\section{Interpretation}

The separate direct/quasi thresholds allow finer control:
\begin{itemize}
    \item Direct identifiers (SSN, MRN, etc.) can have lower thresholds for aggressive protection
    \item Quasi identifiers (age, location, etc.) can have higher thresholds to reduce Wikipedia false positives
    \item k\_safe\_threshold adds a secondary check: even if $k > k_{\min}$, values below k\_safe trigger review
\end{itemize}

\end{document}
"""

    with open(output_dir / "latex_report.tex", "w") as f:
        f.write(latex)
    print("  Saved latex_report.tex")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    input_path: Path,
    output_dir: Path,
) -> None:
    """Run complete analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {input_path}...")
    samples = load_pipeline_results(input_path)
    print(f"  Loaded {len(samples)} samples")

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

    # Baselines
    baseline_tau_review = 0.30
    baseline_tau_block = 0.70
    baseline_k_min = 5
    baseline_k_safe_threshold = 20
    baseline_tau_direct_review = 0.20
    baseline_tau_direct_block = 0.60
    baseline_tau_quasi_review = 0.30
    baseline_tau_quasi_block = 0.70
    baseline_w_direct = 0.50
    baseline_w_quasi = 0.50

    print("Running sweeps...")
    sweeps = run_sweeps(
        samples=samples,
        tau_review_values=tau_review_values,
        tau_block_values=tau_block_values,
        k_min_values=k_min_values,
        k_safe_threshold_values=k_safe_threshold_values,
        tau_direct_review_values=tau_direct_review_values,
        tau_direct_block_values=tau_direct_block_values,
        tau_quasi_review_values=tau_quasi_review_values,
        tau_quasi_block_values=tau_quasi_block_values,
        weight_values=weight_values,
        baseline_tau_review=baseline_tau_review,
        baseline_tau_block=baseline_tau_block,
        baseline_k_min=baseline_k_min,
        baseline_k_safe_threshold=baseline_k_safe_threshold,
        baseline_tau_direct_review=baseline_tau_direct_review,
        baseline_tau_direct_block=baseline_tau_direct_block,
        baseline_tau_quasi_review=baseline_tau_quasi_review,
        baseline_tau_quasi_block=baseline_tau_quasi_block,
        baseline_w_direct=baseline_w_direct,
        baseline_w_quasi=baseline_w_quasi,
    )

    # Compute baseline
    max_direct, max_quasi = compute_normalization(samples)
    baseline = evaluate_operating_point(
        samples=samples,
        max_direct=max_direct,
        max_quasi=max_quasi,
        sweep_name="baseline",
        tau_review=baseline_tau_review,
        tau_block=baseline_tau_block,
        k_min=baseline_k_min,
        k_safe_threshold=baseline_k_safe_threshold,
        tau_direct_review=baseline_tau_direct_review,
        tau_direct_block=baseline_tau_direct_block,
        tau_quasi_review=baseline_tau_quasi_review,
        tau_quasi_block=baseline_tau_quasi_block,
        w_direct=baseline_w_direct,
        w_quasi=baseline_w_quasi,
    )

    # Collect all points and choose operating points
    all_points = []
    for pts in sweeps.values():
        all_points.extend(pts)

    chosen = choose_operating_points(all_points)

    # Create plots
    create_all_plots(sweeps, output_dir)

    # Generate summary
    summary = generate_summary(samples, sweeps, baseline, chosen)

    # Write outputs
    write_json(output_dir / "analysis_summary.json", summary)

    sweep_json = {k: [asdict(p) for p in v] for k, v in sweeps.items()}
    write_json(output_dir / "threshold_sweep_results.json", sweep_json)

    generate_latex_report(summary, output_dir)

    print(f"\nAnalysis complete. Results in {output_dir}/")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\nBaseline (FNR={baseline.false_negative_rate:.3f}, Wiki FPR={baseline.wiki_false_positive_rate:.3f}):")
    print(f"  tau_direct_review={baseline.tau_direct_review}, tau_direct_block={baseline.tau_direct_block}")
    print(f"  tau_quasi_review={baseline.tau_quasi_review}, tau_quasi_block={baseline.tau_quasi_block}")
    print(f"  k_safe_threshold={baseline.k_safe_threshold}")

    for name, point in chosen.items():
        print(f"\n{name}:")
        print(f"  Recall={point.recall:.3f}, FNR={point.false_negative_rate:.3f}, Wiki FPR={point.wiki_false_positive_rate:.3f}")
        print(f"  tau_direct: review={point.tau_direct_review}, block={point.tau_direct_block}")
        print(f"  tau_quasi: review={point.tau_quasi_review}, block={point.tau_quasi_block}")
        print(f"  k_safe_threshold={point.k_safe_threshold}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze privacy router benchmark results with threshold sweeps"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to pipeline_results.json"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("analysis_outputs"),
        help="Output directory for analysis results"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run_analysis(args.input, args.output)


if __name__ == "__main__":
    main()