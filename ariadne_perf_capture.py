"""
Capture perf samples while running warm Ariadne queries.

Strategy:
  1. Open a psycopg connection, pin its backend PID.
  2. Warm up with N queries so all leaf pages cycle through shared_buffers.
  3. Start `perf record -p <pid>` in the background for `duration` seconds.
  4. Drive the same backend with queries continuously during that window.
  5. Wait for perf to flush, then print top hotspots via `perf report`.

Run:
    python3 ariadne_perf_capture.py \\
        --table deep1b_test_l2 \\
        --queries-file /data/vsbt/datasets/deep1b/deep1b_queries.npy \\
        --probes 50,125 --rerank-k 200 \\
        --duration 30
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import psycopg
import pgvector.psycopg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",          default="postgresql://postgres@localhost:5432/postgres")
    ap.add_argument("--table",        required=True)
    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--probes",       default="50,125", help="<outer,inner>")
    ap.add_argument("--rerank-k",     type=int, default=200)
    ap.add_argument("--top",          type=int, default=10)
    ap.add_argument("--warmup",       type=int, default=50)
    ap.add_argument("--duration",     type=int, default=30, help="seconds to capture")
    ap.add_argument("--freq",         type=int, default=999, help="perf sample frequency Hz")
    ap.add_argument("--output",       default="/tmp/ariadne_warm.perf.data")
    args = ap.parse_args()

    po, pi = (int(x) for x in args.probes.split(","))

    conn = psycopg.connect(args.url)
    pgvector.psycopg.register_vector(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT pg_backend_pid()")
        pid = cur.fetchone()[0]
        cur.execute("SET jit = off")
        cur.execute(f"SET ariadne.probes_outer = {po}")
        cur.execute(f"SET ariadne.probes = {pi}")
        cur.execute(f"SET ariadne.rerank_k = {args.rerank_k}")
        # Make sure timing NOTICE doesn't pollute the perf window.
        cur.execute("SET ariadne.log_scan_timings = off")
    print(f"backend pid: {pid}")

    queries = np.load(args.queries_file, mmap_mode="r")
    rng = np.random.default_rng(42)
    q_pool = np.asarray(queries[rng.choice(len(queries), 2000, replace=False)],
                        dtype=np.float32)

    sql = (f"SELECT id FROM {args.table} ORDER BY embedding <-> %s "
           f"LIMIT {args.top}")

    # Warm-up so we're measuring steady-state behavior.
    print(f"warmup: {args.warmup} queries…", flush=True)
    with conn.cursor() as cur:
        for q in q_pool[:args.warmup]:
            cur.execute(sql, (q,))
            cur.fetchall()
    print("warmup done.")

    # Launch perf record in the background. Needs sudo (read other process).
    print(f"starting perf record on pid={pid} for {args.duration}s @ {args.freq}Hz…",
          flush=True)
    perf = subprocess.Popen(
        ["sudo", "perf", "record",
         "-p", str(pid),
         "-F", str(args.freq),
         "-g",
         "--call-graph", "dwarf",
         "-o", args.output,
         "--",
         "sleep", str(args.duration)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Drive queries for the full capture window.
    n_done = 0
    t_end = time.time() + args.duration + 2
    with conn.cursor() as cur:
        while time.time() < t_end:
            q = q_pool[n_done % len(q_pool)]
            cur.execute(sql, (q,))
            cur.fetchall()
            n_done += 1

    perf.wait()
    print(f"ran {n_done} queries during the capture window.")

    conn.close()

    # Make perf.data world-readable so we can analyse without sudo.
    subprocess.run(["sudo", "chown", os.environ.get("USER", "ubuntu"),
                    args.output], check=False)

    print("\n=== Top symbols (self time) ===")
    subprocess.run(
        ["perf", "report", "-i", args.output,
         "--stdio", "--no-children",
         "--sort=symbol,dso",
         "--percent-limit", "0.5"],
        check=False,
    )

    print("\n=== Top symbols (including children) ===")
    subprocess.run(
        ["perf", "report", "-i", args.output,
         "--stdio",
         "--sort=symbol,dso",
         "--percent-limit", "1.0"],
        check=False,
    )


if __name__ == "__main__":
    main()
