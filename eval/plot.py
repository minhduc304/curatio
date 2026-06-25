"""C13 — funnel + benchmark charts. See TDD §7.3.

Reads eval/results/{eval_report,plot_data}.json, writes PNGs to eval/charts/.
Throughput/scaling show the Python worker only until eval/bench.py lands the Rust
numbers (Thu-Fri); they are drawn from BenchResult rows when present.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import settings

RESULTS_DIR = settings.data_dir.parent / "eval" / "results"
CHARTS_DIR = settings.data_dir.parent / "eval" / "charts"


def _load() -> tuple[dict, dict]:
    with open(RESULTS_DIR / "eval_report.json") as f:
        report = json.load(f)
    with open(RESULTS_DIR / "plot_data.json") as f:
        plot_data = json.load(f)
    return report, plot_data


def chart_funnel(report: dict) -> None:
    f = report["funnel"]
    by_stage = f["rejected_by_stage"]
    stages = ["ingested", "heuristic", "quality", "dedup", "kept"]
    remaining = f["ingested"]
    values = [remaining]
    for s in ("heuristic", "quality", "dedup"):
        remaining -= by_stage.get(s, 0)
        values.append(remaining)
    values.append(f["kept"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(stages, values, color="#2b6cb0")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Funnel retention ({f['retention_rate']:.0%} kept)")
    ax.set_ylabel("docs surviving")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_funnel.png", dpi=120)
    plt.close(fig)


def chart_reject_reasons(report: dict) -> None:
    reasons = report["funnel"]["rejected_by_reason"]
    if not reasons:
        return
    items = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)
    labels, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, counts, color="#c05621")
    ax.invert_yaxis()
    ax.set_title("Rejections by reason")
    ax.set_xlabel("docs")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_reject_reasons.png", dpi=120)
    plt.close(fig)


def chart_quality_hist(plot_data: dict) -> None:
    scores = plot_data["quality_scores"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = [i / 20 for i in range(21)]
    ax.hist(scores["neg"], bins=bins, alpha=0.6, label="Command A low (<=2)", color="#c53030")
    ax.hist(scores["pos"], bins=bins, alpha=0.6, label="Command A high (>=4)", color="#2f855a")
    ax.axvline(0.5, color="black", ls="--", lw=1, label="threshold 0.5")
    ax.set_title("Classifier score distribution (held-out)")
    ax.set_xlabel("predicted quality score")
    ax.set_ylabel("docs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_quality_hist.png", dpi=120)
    plt.close(fig)


def chart_dedup_pr(plot_data: dict) -> None:
    curve = plot_data["dedup_pr"]
    thr = [c["threshold"] for c in curve]
    prec = [c["precision"] for c in curve]
    rec = [c["recall"] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thr, prec, "o-", label="precision", color="#2b6cb0")
    ax.plot(thr, rec, "s-", label="recall", color="#c05621")
    ax.axvline(plot_data["operating_threshold"], color="black", ls="--", lw=1, label="operating")
    ax.set_title("Dedup precision/recall vs cosine threshold")
    ax.set_xlabel("cosine threshold")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_dedup_pr.png", dpi=120)
    plt.close(fig)


def chart_throughput(report: dict, plot_data: dict) -> None:
    """Python vs Rust single-worker docs/sec; Rust appears once bench.py runs."""
    benches = {b["runtime"]: b for b in report.get("benchmarks", []) if b["workers"] == 1}
    labels, values = [], []
    if "python" in benches:
        labels.append("python")
        values.append(benches["python"]["docs_per_sec"])
    else:
        labels.append("python (eval)")
        values.append(plot_data["nonapi_docs_per_sec"])
    if "rust" in benches:
        labels.append("rust")
        values.append(benches["rust"]["docs_per_sec"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#2b6cb0", "#dd6b20"][: len(labels)])
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Single-worker non-API throughput")
    ax.set_ylabel("docs/sec")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_throughput.png", dpi=120)
    plt.close(fig)


def chart_scaling(report: dict) -> None:
    """Aggregate throughput vs worker count (Rust); needs bench.py. Skips if absent."""
    rust = sorted(
        (b for b in report.get("benchmarks", []) if b["runtime"] == "rust"),
        key=lambda b: b["workers"],
    )
    if not rust:
        return
    workers = [b["workers"] for b in rust]
    dps = [b["docs_per_sec"] for b in rust]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(workers, dps, "o-", color="#dd6b20")
    ax.set_title("Aggregate throughput vs workers (Rust)")
    ax.set_xlabel("workers")
    ax.set_ylabel("docs/sec")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_scaling.png", dpi=120)
    plt.close(fig)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    report, plot_data = _load()
    chart_funnel(report)
    chart_reject_reasons(report)
    chart_quality_hist(plot_data)
    chart_dedup_pr(plot_data)
    chart_throughput(report, plot_data)
    chart_scaling(report)
    written = sorted(p.name for p in CHARTS_DIR.glob("*.png"))
    print(f"wrote {len(written)} charts to {CHARTS_DIR}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
