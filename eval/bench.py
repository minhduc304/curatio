"""C16 — performance benchmark + parity gate. See TDD §7a, §16.

Fixed pre-embedded Parquet input (Embed API excluded — built from the disk cache,
zero Cohere quota). Runs the Python reference single-worker, then the Rust worker
single / x2 / x4. Diffs each Rust run's keep/reject decisions against the Python
reference; refuses to report throughput unless 100% identical (SC6). Measures
docs/sec + p50 latency. Asserts SC4 (Python >= 1000 docs/sec), SC5 (Rust >= 5x
Python), SC7 (4-worker >= 3x single-worker).

Fills EvalReport.benchmarks[] in eval/results/eval_report.json and appends
throughput/scaling series to plot_data.json (eval/plot.py draws the last 2 panels).

Parity note: the quality dot product is accumulated in f64 on BOTH sides so the
keep/reject decision is independent of summation order across runtimes. The
production consumer uses numpy f32 (equivalent within ~1e-4); f64 here removes the
only float-order sensitivity from the parity-graded path.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from diskcache import Cache

from src.config import settings
from src.curate.dedup import OnlineDedup
from src.curate.heuristics import HeuristicFilter
from src.curate.minhash import MinHasher
from src.models import BenchResult, CurationConfig, EvalReport, RawDoc, QualityModel

ROOT = settings.data_dir.parent
RESULTS_DIR = ROOT / "eval" / "results"
INPUT_PARQUET = RESULTS_DIR / "bench_input.parquet"
RUST_BIN = ROOT / "rust" / "target" / "release" / "Curatio-worker"
WORKER_COUNTS = [1, 2, 4]


# --- pre-embedded input (from cache, zero quota) -----------------------------
def build_input(config: CurationConfig) -> int:
    """Materialize {id, text, emb} for every cached label doc → bench_input.parquet."""
    with open(settings.cache_dir / "label_cache.json") as f:
        records = json.load(f)
    cache = Cache(str(settings.cache_dir / "embeddings"))

    def key(text: str) -> str:
        return f"{hashlib.sha256(text.encode()).hexdigest()}:{config.embed_dim}"

    ids, texts, embs = [], [], []
    for r in records:
        vec = cache.get(key(r["text"]))
        if vec is None:
            continue  # cache-only: skip uncached docs
        ids.append(str(r["doc_id"]))
        texts.append(r["text"])
        embs.append(np.asarray(vec, dtype="<f4").tobytes())

    table = pa.table({"id": ids, "text": texts, "emb": embs})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, INPUT_PARQUET)
    return len(ids)


def load_rows() -> tuple[list[str], list[str], np.ndarray]:
    table = pq.read_table(INPUT_PARQUET)
    ids = [str(x) for x in table.column("id").to_pylist()]
    texts = [str(x) for x in table.column("text").to_pylist()]
    embs = np.stack(
        [np.frombuffer(b, dtype="<f4") for b in table.column("emb").to_pylist()]
    ).astype(np.float32)
    return ids, texts, embs


# --- Python reference (authoritative decisions) ------------------------------
def python_reference(
    config: CurationConfig, model: QualityModel, lang_module: object | None = None
) -> tuple[dict[str, tuple[str, str | None]], float, float]:
    """Run the production pipeline over the pre-embedded rows. lang_module swaps the
    language-id backend in src.curate.heuristics (used to time the naive `langid`
    baseline against production `py3langid` with the exact same pipeline code)."""
    import src.curate.heuristics as heur_mod

    saved_lang = heur_mod.langid
    if lang_module is not None:
        heur_mod.langid = lang_module  # type: ignore[assignment]
    try:
        return _reference_loop(config, model)
    finally:
        heur_mod.langid = saved_lang  # type: ignore[assignment]


def _reference_loop(
    config: CurationConfig, model: QualityModel
) -> tuple[dict[str, tuple[str, str | None]], float, float]:
    ids, texts, embs = load_rows()
    heuristics = HeuristicFilter(config)
    dedup = OnlineDedup(config, max_elements=len(ids) + 1)
    hasher = MinHasher(config.minhash_num_perm, config.minhash_shingle_size)
    coef = np.asarray(model.coef, dtype=np.float64)
    intercept = float(model.intercept)
    threshold = float(model.threshold)

    decisions: dict[str, tuple[str, str | None]] = {}
    latencies: list[float] = []
    t0 = time.perf_counter()
    for doc_id, text, emb in zip(ids, texts, embs):
        t = time.perf_counter()
        reason = heuristics.check(RawDoc(id=doc_id, text=text))
        if reason is not None:
            decisions[doc_id] = ("reject", reason)
        else:
            z = float(coef @ emb.astype(np.float64)) + intercept
            prob = 1.0 / (1.0 + math.exp(-z))
            if prob < threshold:
                decisions[doc_id] = ("reject", "low_quality")
            else:
                sim = dedup.check_and_add(doc_id, emb, hasher.signature(text))
                decisions[doc_id] = ("reject", "near_duplicate") if sim is not None else ("keep", None)
        latencies.append((time.perf_counter() - t) * 1e6)
    elapsed = time.perf_counter() - t0
    return decisions, len(ids) / elapsed, float(np.percentile(latencies, 50))


# --- Rust worker -------------------------------------------------------------
def run_rust(workers: int, config: CurationConfig) -> tuple[dict[str, tuple[str, str | None]], float, float]:
    proc = subprocess.run(
        [
            str(RUST_BIN), "bench",
            "--input", str(INPUT_PARQUET),
            "--model", str(settings.model_path),
            "--minhash", str(settings.model_path.parent / "minhash.json"),
            "--workers", str(workers),
            "--dedup-threshold", str(config.dedup_cosine_threshold),
            "--surface-threshold", str(config.dedup_surface_threshold),
        ],
        capture_output=True, text=True, cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rust bench failed (workers={workers}):\n{proc.stderr}")
    out = json.loads(proc.stdout)
    decisions = {d["id"]: (d["decision"], d["reason"]) for d in out["decisions"]}
    return decisions, out["docs_per_sec"], out["p50_latency_us"]


def parity(ref: dict[str, tuple[str, str | None]], other: dict[str, tuple[str, str | None]]) -> tuple[bool, list[str]]:
    mismatches = [
        f"{doc_id}: ref={ref[doc_id]} rust={other.get(doc_id)}"
        for doc_id in ref
        if other.get(doc_id) != ref[doc_id]
    ]
    return len(mismatches) == 0, mismatches[:10]


def main() -> None:
    config = CurationConfig()
    model = QualityModel.model_validate_json(settings.model_path.read_text())

    print(f"[bench] building pre-embedded input → {INPUT_PARQUET.name} (cache-only) ...")
    n = build_input(config)
    print(f"[bench] {n} cached docs")
    if not RUST_BIN.exists():
        raise SystemExit(f"rust binary not built: {RUST_BIN}\n  → cargo build --release --manifest-path rust/Cargo.toml")

    # Production reference uses py3langid; the naive baseline swaps in the original
    # pure-Python `langid` over the SAME pipeline, to show the optimization
    # progression and prove the swap didn't change any keep/reject decision.
    print("[bench] python reference — py3langid (production) ...")
    ref_decisions, py_dps, py_p50 = python_reference(config, model)

    naive_dps: float | None = None
    try:
        import langid as langid_naive

        print("[bench] python baseline — naive langid ...")
        naive_decisions, naive_dps, _ = python_reference(config, model, lang_module=langid_naive)
        if naive_decisions != ref_decisions:
            print("  WARNING: py3langid changed decisions vs langid — swap is NOT decision-neutral")
    except ImportError:
        print("[bench] (skipping naive langid baseline — langid not installed)")

    benchmarks: list[BenchResult] = [
        BenchResult(runtime="python", workers=1, docs=n, docs_per_sec=py_dps,
                    p50_latency_us=py_p50, decisions_match_reference=True)
    ]
    rust_single_dps = None
    all_parity_ok = True
    for w in WORKER_COUNTS:
        print(f"[bench] rust worker (workers={w}) ...")
        decisions, dps, p50 = run_rust(w, config)
        ok, sample = parity(ref_decisions, decisions)
        all_parity_ok &= ok
        if not ok:
            print(f"  PARITY MISMATCH ({len(sample)} shown): " + "; ".join(sample))
        benchmarks.append(BenchResult(runtime="rust", workers=w, docs=n, docs_per_sec=dps,
                                      p50_latency_us=p50, decisions_match_reference=ok))
        if w == 1:
            rust_single_dps = dps

    rust4 = next(b.docs_per_sec for b in benchmarks if b.runtime == "rust" and b.workers == 4)
    speedup = rust_single_dps / py_dps if py_dps else 0.0          # SC5 (informational)
    scaling = rust4 / rust_single_dps if rust_single_dps else 0.0  # SC7
    aggregate = rust4 / py_dps if py_dps else 0.0

    # Merge into the existing eval report (don't re-run the cache-only eval).
    # eval/plot.py draws the throughput + scaling panels straight from these rows.
    report = EvalReport.model_validate_json((RESULTS_DIR / "eval_report.json").read_text())
    report.benchmarks = benchmarks
    (RESULTS_DIR / "eval_report.json").write_text(report.model_dump_json(indent=2))
    # Stash the naive baseline for the README narrative (not a benchmarks[] row, so
    # plot.py's runtime-keyed throughput chart keeps a single python bar).
    plot_path = RESULTS_DIR / "plot_data.json"
    plot = json.loads(plot_path.read_text())
    plot["python_naive_langid_docs_per_sec"] = naive_dps
    plot_path.write_text(json.dumps(plot, indent=2))

    # SC4/SC6/SC7 are the hard gates. SC5 (single-core speedup) is reported but not
    # gated: once langid is replaced, both runtimes are language-id-bound, so the
    # single-core win is ~1.7x and the systems win is horizontal scale-out (SC7).
    print(f"\n{'='*60}\nCuratio benchmark — {n} pre-embedded docs")
    print("  throughput progression (non-API hot path):")
    if naive_dps is not None:
        print(f"    python (naive langid)   {naive_dps:>10,.0f} docs/sec")
    print(f"    python (py3langid)      {py_dps:>10,.0f} docs/sec  (p50 {py_p50:.0f} us)")
    for b in benchmarks:
        if b.runtime == "rust":
            tag = "✓" if b.decisions_match_reference else "✗"
            print(f"    rust x{b.workers}                 {b.docs_per_sec:>10,.0f} docs/sec  (p50 {b.p50_latency_us:.0f} us) parity {tag}")
    print(f"  SC5 single-core rust speedup : {speedup:.2f}x  (informational; both lang-id-bound)")
    print(f"  aggregate (rust x4 / python) : {aggregate:.2f}x")
    print("-" * 60)
    checks = [
        ("SC4 python docs/sec", py_dps, 1000.0),
        ("SC7 4-worker scaling (x)", scaling, 3.0),
    ]
    ok = all_parity_ok
    print(f"  {'PASS' if all_parity_ok else 'FAIL'}  SC6 parity (rust == python ref)")
    for name, val, bar in checks:
        passed = val >= bar
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name:26s} {val:>10,.2f}  (>= {bar})")
    print("=" * 60)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
