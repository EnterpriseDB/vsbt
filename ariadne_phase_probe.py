"""
Ariadne scan-phase investigation.

For each probe-point in a YAML config:
  - sets ariadne.log_scan_timings = on
  - runs N queries against the existing index
  - parses the per-query NOTICE Ariadne emits
  - aggregates mean / p50 / p95 per phase per probe-point
  - prints a single table

Also runs one EXPLAIN (ANALYZE, BUFFERS) per probe-point so we can see
the executor-side heap-fetch cost for the LIMIT-N result hand-off
(hypothesis #3 — not measured by the scan-time NOTICE).

Use:
    python3 ariadne_phase_probe.py \\
        --url postgresql://postgres@localhost:5432/postgres \\
        --table deep1b_test_l2 \\
        --queries-file /data/vsbt/datasets/deep1b/deep1b_queries.npy \\
        --suite-yaml config/deep1b-test-l2/ariadne-400-160k-b4.yaml \\
        --n-queries 100
"""
import argparse
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
import yaml
import pgvector.psycopg

# Regex to pull metrics out of the ariadne NOTICE. Format (from scan.c):
#  "ariadne scan: total=Tus setup=Sus scan=Cus rerank=Rus | scan-sub: pageio=... prune=... kernel=... unpack=... correct=... heap=... misc=... | ..."
_RE_TOTAL  = re.compile(r"total=([\d.]+)us")
_RE_SETUP  = re.compile(r"setup=([\d.]+)us")
_RE_SCAN   = re.compile(r"\bscan=([\d.]+)us")
_RE_RERANK = re.compile(r"rerank=([\d.]+)us")
_RE_SUB = {
    "pageio":  re.compile(r"pageio=([\d.]+)"),
    "prune":   re.compile(r"prune=([\d.]+)"),
    "kernel":  re.compile(r"kernel=([\d.]+)"),
    "unpack":  re.compile(r"unpack=([\d.]+)"),
    "correct": re.compile(r"correct=([\d.]+)"),
    "heap":    re.compile(r"heap=([\d.]+)"),
    "misc":    re.compile(r"misc=([\d.]+)"),
}


def parse_notice(text):
    if "ariadne scan:" not in text:
        return None
    def grab(rx, t):
        m = rx.search(t)
        return float(m.group(1)) if m else None
    out = {
        "total":  grab(_RE_TOTAL,  text),
        "setup":  grab(_RE_SETUP,  text),
        "scan":   grab(_RE_SCAN,   text),
        "rerank": grab(_RE_RERANK, text),
    }
    for k, rx in _RE_SUB.items():
        out[k] = grab(rx, text)
    return out


def fmt_us(x):
    if x is None:
        return "    -"
    if x < 1000:
        return f"{x:5.0f}us"
    return f"{x/1000:5.1f}ms"


def summarise(samples, keys):
    """Per-key list of (mean, p50, p95) over samples."""
    row = {}
    for k in keys:
        vals = [s[k] for s in samples if s.get(k) is not None]
        if not vals:
            row[k] = (None, None, None)
            continue
        mean = sum(vals) / len(vals)
        p50  = statistics.median(vals)
        p95  = sorted(vals)[max(0, int(len(vals)*0.95) - 1)]
        row[k] = (mean, p50, p95)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",          required=True)
    ap.add_argument("--table",        required=True)
    ap.add_argument("--queries-file", required=True, help=".npy of query vectors")
    ap.add_argument("--suite-yaml",   required=True)
    ap.add_argument("--n-queries",    type=int, default=100)
    ap.add_argument("--top",          type=int, default=10)
    args = ap.parse_args()

    with open(args.suite_yaml) as f:
        suite_cfg = yaml.safe_load(f)
    suite_name = next(iter(suite_cfg))
    cfg = suite_cfg[suite_name]
    rerank_k = cfg.get("rerank_k", 0)
    metric = cfg.get("metric", "l2")
    op = "<->" if metric == "l2" else ("<#>" if metric == "ip" else "<=>")

    queries = np.load(args.queries_file, mmap_mode="r")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(queries), size=args.n_queries, replace=False)
    q_samples = np.asarray(queries[idx], dtype=np.float32)

    benchmarks = cfg["benchmarks"]

    rows_summary = []
    for name, bench in benchmarks.items():
        probes_pair = bench["probes"]
        po, pi = (int(x) for x in str(probes_pair).split(","))

        notices = []
        # Per-process psycopg connection; install pgvector + a notice handler.
        conn = psycopg.connect(args.url)
        pgvector.psycopg.register_vector(conn)
        def on_notice(diag, _captured=notices):
            _captured.append(diag.message_primary or "")
        conn.add_notice_handler(on_notice)

        with conn.cursor() as cur:
            cur.execute("SET jit = off")
            cur.execute(f"SET ariadne.probes_outer = {po}")
            cur.execute(f"SET ariadne.probes = {pi}")
            cur.execute(f"SET ariadne.rerank_k = {rerank_k}")
            cur.execute("SET ariadne.log_scan_timings = on")

        sql_idx_scan = (
            f"SELECT id FROM {args.table} ORDER BY embedding {op} %s LIMIT {args.top}"
        )

        # 1) Phase-NOTICE pass
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            for q in q_samples:
                cur.execute(sql_idx_scan, (q,))
                _ = cur.fetchall()
        wall_total = time.perf_counter() - t0
        qps = len(q_samples) / wall_total if wall_total > 0 else 0.0

        parsed = [p for p in (parse_notice(n) for n in notices) if p]

        # 2) Single EXPLAIN ANALYZE BUFFERS — hypothesis #3
        with conn.cursor() as cur:
            cur.execute("SET ariadne.log_scan_timings = off")
            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql_idx_scan}",
                        (q_samples[0],))
            plan = "\n".join(row[0] for row in cur.fetchall())

        conn.close()

        keys = ["total", "setup", "scan", "rerank",
                "pageio", "prune", "kernel", "unpack", "correct", "heap", "misc"]
        agg = summarise(parsed, keys)

        # Pull executor-side heap-fetch cost from EXPLAIN BUFFERS
        m_limit = re.search(r"Limit.*?actual time=([\d.]+)\.\.([\d.]+)", plan, re.S)
        m_idx   = re.search(r"Index Scan.*?actual time=([\d.]+)\.\.([\d.]+)", plan, re.S)
        m_bufs  = re.search(r"Index Scan.*?Buffers:\s*shared\s+hit=(\d+)(?:\s+read=(\d+))?",
                            plan, re.S)
        limit_end = float(m_limit.group(2)) if m_limit else None
        idx_end   = float(m_idx.group(2))   if m_idx   else None
        heap_fetch_ms = (limit_end - idx_end) if (limit_end and idx_end) else None
        buf_hit  = int(m_bufs.group(1)) if m_bufs else None
        buf_read = int(m_bufs.group(2)) if (m_bufs and m_bufs.group(2)) else 0

        rows_summary.append((name, po, pi, qps, len(parsed), agg,
                             heap_fetch_ms, buf_hit, buf_read))

        print(f"\n=== probe {name} ({po},{pi})  QPS={qps:.1f}  samples={len(parsed)} ===")
        hdr = f"{'phase':<10} {'mean':>9} {'p50':>9} {'p95':>9}"
        print(hdr)
        print("-" * len(hdr))
        for k in keys:
            mean, p50, p95 = agg[k]
            print(f"{k:<10} {fmt_us(mean):>9} {fmt_us(p50):>9} {fmt_us(p95):>9}")
        if heap_fetch_ms is not None:
            print(f"executor heap-fetch (LIMIT-{args.top}) ≈ {heap_fetch_ms:.2f}ms"
                  f"   buffers hit={buf_hit} read={buf_read}")

    # Compact summary table at the end
    print("\n" + "=" * 92)
    print("SUMMARY — mean per phase, in milliseconds (per query)")
    print("=" * 92)
    cols = ["probes", "QPS", "total", "setup", "scan", "rerank",
            "kernel", "pageio", "heap_sub", "heap_exec"]
    print("{:<10} {:>6}".format("probes", "QPS")
          + "".join(f" {c:>9}" for c in cols[2:]))
    for (name, po, pi, qps, _n, agg, hf_ms, _hit, _rd) in rows_summary:
        def m(k):
            v = agg[k][0]
            return "    -" if v is None else f"{v/1000:6.2f}"
        hf = "    -" if hf_ms is None else f"{hf_ms:6.2f}"
        print(f"{po:>3},{pi:<5} {qps:>6.1f}  {m('total')}    {m('setup')}    "
              f"{m('scan')}    {m('rerank')}    {m('kernel')}    {m('pageio')}    "
              f"{m('heap')}    {hf}")


if __name__ == "__main__":
    main()
