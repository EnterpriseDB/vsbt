"""
Run a sweep of query-time GUC variants against a pre-built Ariadne
index, measuring recall@k and QPS at fixed probe points.

The variants probe three open questions:
  1. Does the §5.3 certified lower-bound improve recall? (→ keep
     factor_err_fp16 in the on-disk header, or drop in v13)
  2. Is our quantized routing tape the per-probe-recall gap vs VC?
  3. Are there cheap query-time knobs that close the gap?

No rebuilds. Runs on whatever index is currently bound to `--table`.

Usage:
    python3 ariadne_variant_sweep.py \\
        --url postgresql://postgres@localhost:5432/postgres \\
        --table deep1b_test_l2 \\
        --queries-file /data/vsbt/datasets/deep1b/deep1b_queries.npy \\
        --ground-truth /data/vsbt/datasets/deep1b/deep1b_groundtruth.npy \\
        --probes "50,125" "80,200" \\
        --n-queries 1000
"""
import argparse
import statistics
import sys
import time

import numpy as np
import psycopg
import pgvector.psycopg


# Each variant is (label, GUC settings dict). Settings missing from a
# variant fall back to defaults (which we reset before each variant).
# Defaults — all set before every run so variants are isolated.
DEFAULTS = {
    "ariadne.lowerbound_mode":    "legacy",
    "ariadne.use_bit_routing":    "on",
    "ariadne.routing_refine_mult": "5",
    # rerank_k + probe_outer + probes are set per-run.
}

VARIANTS = [
    ("baseline",         {}),  # defaults
    ("lb_static",        {"ariadne.lowerbound_mode": "static"}),
    ("lb_calibrated",    {"ariadne.lowerbound_mode": "calibrated"}),
    ("lb_adaptive",      {"ariadne.lowerbound_mode": "adaptive"}),
    ("no_bit_routing",   {"ariadne.use_bit_routing": "off"}),
    ("refine_mult_10",   {"ariadne.routing_refine_mult": "10"}),
    ("refine_mult_20",   {"ariadne.routing_refine_mult": "20"}),
    ("rerank_400",       {"_rerank_k": 400}),  # query-side override
]


def apply_settings(conn, settings):
    """Apply DEFAULTS first, then overlay variant settings. _* keys are skipped."""
    merged = dict(DEFAULTS)
    for k, v in settings.items():
        if not k.startswith("_"):
            merged[k] = v
    with conn.cursor() as cur:
        for k, v in merged.items():
            cur.execute(f"SET {k} = {repr(str(v))}")


def run_variant(conn, table, op, top, queries, ground_truth,
                probes_outer, probes_inner, rerank_k):
    """Run all queries, return (recall, qps, p50_ms, p95_ms)."""
    with conn.cursor() as cur:
        cur.execute(f"SET ariadne.probes_outer = {probes_outer}")
        cur.execute(f"SET ariadne.probes       = {probes_inner}")
        cur.execute(f"SET ariadne.rerank_k     = {rerank_k}")
        cur.execute("SET jit = off")

    sql = f"SELECT id FROM {table} ORDER BY embedding {op} %s LIMIT {top}"

    latencies = []
    hits = 0
    n = 0
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        for q, gt in zip(queries, ground_truth):
            ts = time.perf_counter()
            cur.execute(sql, (q,))
            rows = cur.fetchall()
            latencies.append((time.perf_counter() - ts) * 1000.0)
            result_ids = {r[0] for r in rows[:top]}
            gt_ids = set(gt[:top].tolist())
            hits += len(result_ids & gt_ids)
            n += 1
    wall = time.perf_counter() - t0
    recall = hits / (n * top)
    qps = n / wall
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    return recall, qps, p50, p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",          default="postgresql://postgres@localhost:5432/postgres")
    ap.add_argument("--table",        required=True)
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--metric",       default="l2", choices=["l2", "ip", "cosine"])
    ap.add_argument("--top",          type=int, default=10)
    ap.add_argument("--rerank-k",     type=int, default=200,
                    help="default rerank_k; overridden by rerank_400 variant")
    ap.add_argument("--probes",       nargs="+", default=["50,125", "80,200"],
                    help="probe pairs to test")
    ap.add_argument("--n-queries",    type=int, default=1000)
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    op = {"l2": "<->", "ip": "<#>", "cosine": "<=>"}[args.metric]

    print(f"loading queries from {args.queries_file}", flush=True)
    queries = np.load(args.queries_file, mmap_mode="r")
    ground_truth = np.load(args.ground_truth, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(queries), size=args.n_queries, replace=False)
    q_sample = np.asarray(queries[idx], dtype=np.float32)
    gt_sample = np.asarray(ground_truth[idx], dtype=np.int64)
    print(f"  {len(q_sample)} queries sampled (seed={args.seed})", flush=True)

    conn = psycopg.connect(args.url)
    pgvector.psycopg.register_vector(conn)

    # Probe pairs come in as "po,pi"
    probe_pairs = [tuple(int(x) for x in p.split(",")) for p in args.probes]

    results = []  # rows: (variant, po, pi, recall, qps, p50, p95)

    for label, settings in VARIANTS:
        rerank_k = settings.get("_rerank_k", args.rerank_k)
        apply_settings(conn, settings)
        for po, pi in probe_pairs:
            print(f"  running {label:18} @ ({po},{pi}) rerank_k={rerank_k} …",
                  flush=True)
            recall, qps, p50, p95 = run_variant(
                conn, args.table, op, args.top,
                q_sample, gt_sample,
                po, pi, rerank_k,
            )
            results.append((label, po, pi, rerank_k, recall, qps, p50, p95))
            print(f"     recall={recall:.4f} qps={qps:.1f} p50={p50:.2f} ms"
                  f" p95={p95:.2f} ms", flush=True)

    conn.close()

    # ---- Summary tables, one per probe pair -----------------------------
    for po, pi in probe_pairs:
        print()
        print(f"=== Summary @ probes=({po},{pi}) ===")
        rows = [r for r in results if r[1] == po and r[2] == pi]
        base = next(r for r in rows if r[0] == "baseline")
        base_recall, base_qps = base[4], base[5]
        hdr = f"{'variant':<18} {'rerank_k':>9} {'recall':>8} {'Δrec':>8} {'QPS':>7} {'ΔQPS%':>7} {'p50ms':>7} {'p95ms':>7}"
        print(hdr)
        print("-" * len(hdr))
        for (label, _, _, rk, recall, qps, p50, p95) in rows:
            drecall = recall - base_recall
            dqps_pct = 100.0 * (qps - base_qps) / base_qps if base_qps > 0 else 0.0
            print(f"{label:<18} {rk:>9d} {recall:>8.4f} "
                  f"{drecall:+8.4f} {qps:>7.1f} {dqps_pct:+7.1f}% "
                  f"{p50:>7.2f} {p95:>7.2f}")

    # ---- Verdicts -------------------------------------------------------
    print()
    print("=== Verdicts ===")
    # 1) factor_err utility: best lowerbound vs legacy at each probe pair
    for po, pi in probe_pairs:
        rows = {r[0]: r for r in results if r[1] == po and r[2] == pi}
        base_recall = rows["baseline"][4]
        lb_best = max(rows[k][4] for k in ("lb_static", "lb_calibrated", "lb_adaptive"))
        delta = lb_best - base_recall
        verdict = ("KEEP factor_err_fp16" if delta >= 0.005
                   else "drop factor_err_fp16 in v13")
        print(f"  @ ({po},{pi}): best lowerbound = {lb_best:.4f}, "
              f"Δ={delta:+.4f} → {verdict}")
    # 2) bit-routing impact
    for po, pi in probe_pairs:
        rows = {r[0]: r for r in results if r[1] == po and r[2] == pi}
        delta_recall = rows["no_bit_routing"][4] - rows["baseline"][4]
        verdict = ("bit-routing is leaking recall — investigate"
                   if delta_recall >= 0.005
                   else "bit-routing is fine (no recall left on the table)")
        print(f"  @ ({po},{pi}): no_bit_routing Δrecall={delta_recall:+.4f} "
              f"→ {verdict}")
    # 3) refine_mult sweep
    for po, pi in probe_pairs:
        rows = {r[0]: r for r in results if r[1] == po and r[2] == pi}
        d10 = rows["refine_mult_10"][4] - rows["baseline"][4]
        d20 = rows["refine_mult_20"][4] - rows["baseline"][4]
        verdict = ("bump default refine_mult"
                   if max(d10, d20) >= 0.005 else
                   "current refine_mult=5 is sufficient")
        print(f"  @ ({po},{pi}): refine_mult 10/20 Δrecall={d10:+.4f}/{d20:+.4f} "
              f"→ {verdict}")
    # 4) rerank_k bump
    for po, pi in probe_pairs:
        rows = {r[0]: r for r in results if r[1] == po and r[2] == pi}
        delta = rows["rerank_400"][4] - rows["baseline"][4]
        verdict = ("bump default rerank_k" if delta >= 0.005
                   else "rerank_k=200 is enough")
        print(f"  @ ({po},{pi}): rerank_400 Δrecall={delta:+.4f} → {verdict}")


if __name__ == "__main__":
    main()
