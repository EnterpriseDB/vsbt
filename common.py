import argparse
import gc
import multiprocessing as mp
import os
import re
import threading
import time

import numpy as np
import psutil
import psycopg
import yaml
from tqdm import tqdm

import datasets
from loader import load_vectors, load_vectors_generator
from monitor import (
    SystemMonitor,
    PGStatsCollector,
    generate_system_report,
    is_local_database,
    detect_pg_io_device,
)


def build_arg_parse(parser: argparse.ArgumentParser):
    """Common arguments for all benchmarks."""

    parser.add_argument(
        "-s", "--suite",
        help="The YAML file containing the test suite configuration",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--url",
        help="URL, like `postgresql://postgres@localhost:5432/postgres`",
        default="postgresql://postgres@localhost:5432/postgres",
        required=False,
    )

    parser.add_argument(
        "--devices",
        nargs='+',
        help='Block devices to be monitored (auto-detected if not specified)',
        default=None,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size for loading the dataset from the hdf5 file",
        default=1000000,
    )

    parser.add_argument(
        "-c", "--centroids-file",
        help="Path to the centroids file",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-ct", "--centroids-table",
        help="Centroids table name",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--skip-add-embeddings",
        help="Skip adding embeddings",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--skip-index-creation",
        help="Skip index creation step",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--query-clients",
        type=int,
        help="Number of parallel client sessions for querying",
        default=1,
        required=False,
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        help="Limit the number of queries per benchmark (default: use full test set)",
        default=None,
        required=False,
    )

    parser.add_argument(
        "--debug",
        help="Enable debug logging",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--max-load-threads",
        type=int,
        help="Maximum number of parallel threads for loading embeddings",
        default=4,  # Default value for max_load_threads
        required=False,
    )

    parser.add_argument(
        "--overwrite-table",
        help="Overwrite existing table when adding embeddings",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--debug-single-query",
        help="Debug mode: repeat the same query to diagnose latency degradation",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--build-only",
        help="Only build the index, skip query benchmarks",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--warmup",
        type=str,
        default="auto",
        help=(
            "Cache warmup mode: 'auto' (prewarm index + adaptive query warmup "
            "per benchmark point, default), 'off' (skip both prewarm and "
            "warmup queries), or an integer N (prewarm + run exactly N warmup "
            "queries per benchmark point). Warmup queries do not count toward "
            "--max-queries."
        ),
        required=False,
    )


# Warmup knee detector: stop when the trailing N-chunk mean QPS no longer
# trends. Absolute latency-CV thresholds would never trip on HNSW because
# per-query traversal variance is ~30% even with fully warm caches.
WARMUP_CHUNK = 50
WARMUP_CHUNKS_PER_WINDOW = 4
WARMUP_QPS_TOLERANCE = 0.05
WARMUP_MIN = 200
WARMUP_MAX = 5000
WARMUP_HARD_FLOOR = 100


def psql_log_handler(diag):
    """Notice handler for psycopg connections — prints PostgreSQL NOTICEs."""
    if diag.severity and diag.message_primary:
        print(f"[PG {diag.severity}] {diag.message_primary}")


def get_keepalive_kwargs() -> dict:
    """Get keepalive arguments for the database connection."""

    return {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 5,
        "keepalives_count": 5,
    }


def load_suite_config(suite_file: str) -> dict:
    """Load the test suite configuration from a YAML file."""

    config = {}
    with open(suite_file, "r") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(
            f"configuration file {suite_file} is empty or invalid")

    return config

def calculate_coverage(time_intervals):
    if not time_intervals:
        return 0
    sorted_intervals = sorted(time_intervals, key=lambda x: x[0])
    merged = []
    current_start, current_end = sorted_intervals[0]
    for interval in sorted_intervals[1:]:
        next_start, next_end = interval
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    total_length = 0
    for start, end in merged:
        total_length += end - start
    return total_length


def calculate_metrics(
    all_results,
    k,
    m,
    query_clients,
) -> tuple[float, float, float, float]:
    """Calculate recall, QPS, and latency percentiles from results"""
    hits, latencies = zip(*all_results)

    if isinstance(latencies[0], (list, tuple)):
        # parallel_bench: calculating coverage of latencies
        total_time = calculate_coverage(latencies)
        latencies = [(end - start) for start, end in latencies]
    else:
        # sequential_bench: calculating total time of latencies
        total_time = sum(latencies)

    total_hits = sum(hits)

    recall = total_hits / (k * m * query_clients)
    qps = (m * query_clients) / total_time

    # Calculate latency percentiles (in milliseconds)
    latencies_ms = np.array(latencies) * 1000
    p50 = np.percentile(latencies_ms, 50)
    p99 = np.percentile(latencies_ms, 99)

    return recall, qps, p50, p99


class TestSuite:
    """Base class for test suites."""

    keepalive_kwargs = get_keepalive_kwargs()

    def __init__(self,
                 suite_file: str,
                 url: str,
                 devices,
                 chunk_size: int,
                 skip_add_embeddings: bool = False,
                 centroids: str = None,
                 centroids_table: str = None,
                 skip_index_creation: bool = False,
                 query_clients: int = 1,
                 max_load_threads: int = 4,
                 debug: bool = False,
                 overwrite_table: bool = False,
                 debug_single_query: bool = False,
                 build_only: bool = False,
                 max_queries: int = None,
                 warmup: str = "auto"):
        self.suite_file = suite_file
        self.config = load_suite_config(suite_file)
        self.url = url
        self.devices = devices
        self.chunk_size = chunk_size
        self.skip_add_embeddings = skip_add_embeddings
        self.skip_index_creation = skip_index_creation
        self.centroids = centroids
        self.centroids_table = centroids_table
        self.results = {}
        self.query_clients = query_clients
        self.max_load_threads = max_load_threads
        self.debug = debug
        self.overwrite_table = overwrite_table
        self.debug_single_query = debug_single_query
        self.build_only = build_only
        self.max_queries = max_queries
        self.warmup = self._parse_warmup(warmup)
        self._warmup_q_stats = {}

        # Check if database is local or remote
        self.is_local_db = is_local_database(url)
        if not self.is_local_db:
            print(f"Remote database detected. System monitoring will be skipped.")
            print(f"PostgreSQL statistics will still be collected.")

        # Monitors (initialized per suite)
        self.system_monitor = None
        self.pg_stats_collector = None

    def debug_log(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def init_ext(self, suite_name: str = None):
        raise NotImplementedError(
            "init_ext should be implemented in subclasses.")

    def create_connection(self) -> psycopg.Connection:
        """Create a database connection."""
        conn = psycopg.Connection.connect(
            conninfo=self.url,
            autocommit=True,
            **self.keepalive_kwargs,
        )
        return conn

    def make_batch_args(self, test, answer, top, metric_ops, table_name, benchmark,
                        warmup_n=0):
        return test, answer, top, metric_ops, table_name, warmup_n

    def apply_session_guc(self, conn, benchmark):
        """Apply per-benchmark session GUCs (probes/ef_search/etc).

        Overridden per suite; called by the parallel warmup probe so the
        warmup loop exercises the same code path as the measurement.
        """
        return

    @staticmethod
    def _parse_warmup(warmup):
        if warmup is None:
            return "auto"
        if isinstance(warmup, int):
            if warmup < 0:
                raise ValueError(f"--warmup integer must be >= 0, got {warmup}")
            return warmup
        s = str(warmup).strip().lower()
        if s in ("auto", "off"):
            return s
        try:
            n = int(s)
        except ValueError:
            raise ValueError(
                f"--warmup must be 'auto', 'off', or an integer; got {warmup!r}"
            )
        if n < 0:
            raise ValueError(f"--warmup integer must be >= 0, got {n}")
        return n

    def warmup_query(self, table_name, dataset, metric_ops, top, benchmark):
        """Return (sql, bind_fn) for warmup and measurement queries.

        Default is the single-stage SQL with a 1-arg bind. Subclasses
        with multi-stage queries (e.g. IVFFlat-BQ + rerank) override this
        so the warmup loop exercises the same plan as the measurement
        loop. The `benchmark` dict carries per-point params (e.g. a
        rerank amplification factor) when the override needs them.
        """
        sql = (
            f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s "
            f"LIMIT {top}"
        )
        return sql, (lambda q: (q,))

    def warmup_for_benchmark(self, conn, table_name, dataset, metric_ops, top,
                             benchmark_name, benchmark=None):
        """Run warmup queries on `conn`; returns (n_warmup_queries, converged).

        The caller must apply per-benchmark GUCs on `conn` BEFORE invoking
        this method. Warmup queries iterate dataset["test"] with wrap-around;
        results and latencies are discarded.
        """
        if self.warmup == "off":
            return 0, False
        if self.debug_single_query:
            # Repeating the same query is the whole point of that mode —
            # warmup would defeat it.
            return 0, False

        test = dataset["test"]
        n_test = test.shape[0] if hasattr(test, "shape") else len(test)
        if n_test == 0:
            return 0, False

        query_sql, bind_fn = self.warmup_query(
            table_name, dataset, metric_ops, top, benchmark
        )

        cursor = conn.cursor()
        latencies = []
        converged = False

        if isinstance(self.warmup, int):
            target = self.warmup
            min_n = target
            max_n = target
            check_cv = False
        else:
            target = WARMUP_MAX
            min_n = WARMUP_MIN
            max_n = WARMUP_MAX
            check_cv = True

        i = 0
        chunk_qps_history = []
        chunk_start_time = time.perf_counter()
        chunk_count_at_start = 0
        K = WARMUP_CHUNKS_PER_WINDOW
        try:
            while i < max_n:
                q = test[i % n_test]
                t0 = time.perf_counter()
                cursor.execute(query_sql, bind_fn(q))
                cursor.fetchall()
                latencies.append(time.perf_counter() - t0)
                i += 1

                if not check_cv:
                    continue
                if i % WARMUP_CHUNK != 0:
                    continue
                now = time.perf_counter()
                chunk_dt = now - chunk_start_time
                chunk_n = i - chunk_count_at_start
                chunk_qps = chunk_n / chunk_dt if chunk_dt > 0 else float("nan")
                chunk_qps_history.append(chunk_qps)
                chunk_start_time = now
                chunk_count_at_start = i

                drift = float("nan")
                if len(chunk_qps_history) >= 2 * K:
                    recent = chunk_qps_history[-K:]
                    prior = chunk_qps_history[-2 * K:-K]
                    recent_mean = sum(recent) / K
                    prior_mean = sum(prior) / K
                    if prior_mean > 0:
                        drift = abs(recent_mean - prior_mean) / prior_mean

                if self.debug:
                    print(
                        f"[warmup-trace] {benchmark_name} n={i} "
                        f"chunk_qps={chunk_qps:.1f} drift={drift:.4f}"
                    )

                if i < max(min_n, WARMUP_HARD_FLOOR):
                    continue
                if not (drift == drift):  # NaN guard
                    continue
                if drift < WARMUP_QPS_TOLERANCE:
                    converged = True
                    break
        finally:
            cursor.close()

        capped = (not converged and check_cv and i >= max_n)
        if latencies:
            ms = [x * 1000 for x in latencies]
            ms_sorted = sorted(ms)
            p50 = ms_sorted[len(ms_sorted) // 2]
            total_s = sum(latencies)
            qps = i / total_s if total_s > 0 else float("nan")
            tag = "converged" if converged else ("capped" if capped else "fixed")
            print(
                f"[warmup] {benchmark_name}: n={i} {tag} "
                f"p50={p50:.2f}ms qps={qps:.2f}"
            )
            self._warmup_q_stats[benchmark_name] = {
                "n": i,
                "converged": converged,
                "capped": capped,
                "p50_ms": p50,
                "qps": qps,
            }
        return i, converged

    def check_index_fits_shared_buffers(self, conn, index_name: str, table_name: str):
        """Check if the index fits in shared_buffers and print size summary."""
        try:
            row = conn.execute(
                "SELECT pg_relation_size(%s) AS idx_size, "
                "pg_total_relation_size(%s) AS tbl_size, "
                "setting::bigint * 8192 AS sb_size "
                "FROM pg_settings WHERE name = 'shared_buffers'",
                (index_name, table_name),
            ).fetchone()
            if row:
                idx_size, tbl_size, sb_size = row
                idx_gb = idx_size / (1024 ** 3)
                tbl_gb = tbl_size / (1024 ** 3)
                sb_gb = sb_size / (1024 ** 3)
                coverage = min(100.0, 100.0 * sb_size / idx_size) if idx_size > 0 else 0
                partial = " (prewarm will be partial)" if idx_size > sb_size else ""
                print(f"Table: {tbl_gb:.1f} GB | Index: {idx_gb:.1f} GB | "
                      f"shared_buffers: {sb_gb:.1f} GB (index coverage: {coverage:.1f}%){partial}")
        except psycopg.Error:
            pass

    def prewarm_index(self, table_name: str):
        raise NotImplementedError("prewarm_index should be implemented in subclasses.")

    def add_embeddings(self, suite_name: str, table_name: str, ds: dict, pg_parallel_workers: int = None):
        """
        Add embeddings to the database.
        Handles HDF5, NPY (mmap), and generator data sources.
        """
        dim = ds["dim"]
        n = ds["num"]
        data = ds["train"]

        is_sliceable = hasattr(data, "__getitem__") and hasattr(data, "shape")
        conn = self.create_connection()

        if self.debug:
            print(f"\n📦 Load Configuration:")
            print(f"    • Table:           {table_name}")
            print(f"    • Rows:            {n:,}")
            print(f"    • Dimensions:      {dim}")
            print(f"    • Load Threads:    {self.max_load_threads}")
            print(f"    • Chunk Size:      {self.chunk_size:,}")
            print(f"    • Overwrite:       {self.overwrite_table}")
            print()

        # --- Table Setup ---
        if self.overwrite_table:
            print(f"Dropping existing table {table_name}...", end="", flush=True)
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            print(" done!")

        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            table_exists = True
        except psycopg.errors.UndefinedTable:
            table_exists = False

        if table_exists:
            print(f"Table {table_name} already exists, using it")
            conn.close()
            return

        conn.execute(f"CREATE TABLE {table_name} (id integer, embedding vector({dim}))")
        if pg_parallel_workers is not None:
            conn.execute(f"ALTER TABLE {table_name} SET (parallel_workers = {pg_parallel_workers})")
        conn.commit()
        conn.close()

        # --- Loading Data ---
        start_time = time.perf_counter()
        chunk_size = self.chunk_size if self.chunk_size else 100_000
        num_threads = self.max_load_threads or 1

        pbar = tqdm(desc="Adding embeddings", total=n, unit=" rows", ncols=80, unit_scale=True)

        if is_sliceable:
            load_vectors(
                conn_factory=self.create_connection,
                table_name=table_name,
                data=data,
                n=n,
                chunk_size=chunk_size,
                num_threads=num_threads,
                progress=pbar,
            )
        else:
            conn = self.create_connection()
            load_vectors_generator(conn, table_name, data, n, progress=pbar)
            conn.close()

        pbar.close()

        end_time = time.perf_counter()
        self.results[suite_name]["load_time"] = int(round(end_time - start_time))
        print(f'Dataset load time: {self.results[suite_name]["load_time"]}s')

    def add_centroids_to_table(self, centroids_file: str, table_name: str = "public.centroids"):
        # Load centroids from the .npy file
        centroids = np.load(centroids_file)
        if centroids.ndim != 2:
            raise ValueError("Centroids file must contain a 2D array.")

        vector_dimensions = centroids.shape[1]
        conn = self.create_connection()

        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"""
            CREATE TABLE {table_name} (
                id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                parent INT,
                vector vector({vector_dimensions})
            )
        """)

        if centroids.size == 0:
            print("\tNo centroids to store, skipping")
            return

        root = centroids[0]
        with conn.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {table_name} (id, parent, vector) OVERRIDING SYSTEM VALUE VALUES (0, NULL, %s)",
                (root,)
            )

        with conn.cursor() as cursor:
            for centroid in centroids[1:]:
                cursor.execute(
                    f"INSERT INTO {table_name} (parent, vector) VALUES (0, %s)",
                    (centroid,)
                )
        self.debug_log(f"Imported {len(centroids)} centroids into {table_name}.")

    @staticmethod
    def _parse_phase(raw_phase: str):
        """Strip embedded percentage from phase names like '... (30 %)'.

        Returns (base_phase, pct) where pct is an int 0-100 or None.
        """
        m = re.match(r'^(.*?)\s*\((\d+)\s*%\)\s*$', raw_phase)
        if m:
            return m.group(1).rstrip(), int(m.group(2))
        return raw_phase, None

    def monitor_index_build(self, event: threading.Event):
        conn = self.create_connection()
        with conn.cursor() as acur:
            phase = "initializing"
            pbar = None
            print("Building index (initializing)...", flush=True)

            while True:
                if event.is_set():
                    if pbar is not None:
                        pbar.update(pbar.total - pbar.n)
                        pbar.close()
                    conn.close()
                    break

                acur.execute("SELECT blocks_done, blocks_total, phase FROM pg_stat_progress_create_index")
                result = acur.fetchone()

                if result:
                    blocks_done = result[0] or 0
                    new_total = result[1] or 0
                    raw_phase = result[2] or phase
                    base_phase, embedded_pct = self._parse_phase(raw_phase)

                    if base_phase != phase:
                        if pbar is not None:
                            pbar.close()
                            pbar = None
                        phase = base_phase

                        if embedded_pct is None and new_total == 0:
                            print(f"Building index ({phase})...", flush=True)

                    if embedded_pct is not None:
                        if pbar is None:
                            pbar = tqdm(smoothing=0.0, total=100,
                                        desc=f"Building index ({phase})", ncols=100,
                                        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]")
                        pbar.update(max(embedded_pct - pbar.n, 0))
                    elif new_total > 0:
                        if pbar is None:
                            pbar = tqdm(smoothing=0.0, total=new_total, initial=blocks_done,
                                        desc=f"Building index ({phase})", ncols=100,
                                        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]")
                        elif new_total != pbar.total:
                            pbar.total = new_total
                            pbar.n = 0
                            pbar.refresh()
                        pbar.update(max(blocks_done - pbar.n, 0))

                time.sleep(0.5)

    def index_name(self, table_name: str) -> str:
        return f"{table_name}_embedding_idx"

    def create_index(self, suite_name: str, table_name: str, dataset: dict) -> tuple[
        threading.Event, threading.Thread]:
        os.makedirs(f"./results/{suite_name}/index_build", exist_ok=True)
        conn = self.create_connection()
        idx_name = self.index_name(table_name)
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (idx_name,))
            if cur.fetchone():
                print(f"Dropping index {idx_name}...", end="", flush=True)
                try:
                    conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
                    print(" done!")
                except Exception:
                    print(" failed!")

        pg_parallel_workers = self.config[suite_name].get("pg_parallel_workers")
        if pg_parallel_workers is not None:
            conn.execute(f"ALTER TABLE {table_name} SET (parallel_workers = {pg_parallel_workers})")
        else:
            conn.execute(f"ALTER TABLE {table_name} RESET (parallel_workers)")

        conn.close()

        event = threading.Event()

        index_monitor_thread = threading.Thread(
            target=self.monitor_index_build,
            args=(event,),
        )
        index_monitor_thread.start()

        return event, index_monitor_thread

    def calculate_index_size(self, suite_name: str, table_name: str):
        conn = self.create_connection()
        idx_name = self.index_name(table_name)
        with conn.cursor() as acur:
            acur.execute(
                "SELECT pg_size_pretty(pg_relation_size(%s::regclass))",
                (idx_name,)
            )
            result = acur.fetchone()
            self.results[suite_name]["index_size"] = result[0]
        conn.close()

    def verify_index_usage(self, table_name: str, dataset: dict, metric_ops: str, top: int, benchmark: dict):
        """Run EXPLAIN on the search query and warn if no index scan is detected."""
        query_sql, bind_fn = self.warmup_query(table_name, dataset, metric_ops, top, benchmark)
        sample_query = dataset["test"][0]
        conn = self.create_connection()
        try:
            self.apply_session_guc(conn, benchmark)
            rows = conn.execute(f"EXPLAIN {query_sql}", bind_fn(sample_query)).fetchall()
            plan_text = "\n".join(row[0] for row in rows)
            plan_lower = plan_text.lower()
            if "index scan" not in plan_lower and "index only scan" not in plan_lower:
                print(f"WARNING: query plan does not use an index scan:")
                print(plan_text)
            else:
                self.debug_log("EXPLAIN verified: query uses index scan")
                if self.debug:
                    print(plan_text)
        except Exception as e:
            self.debug_log(f"EXPLAIN verification failed: {e}")
        finally:
            conn.close()

    def sequential_bench(self, name: str, table_name: str, conn: psycopg.Connection, metric_ops: str, top: int,
                         benchmark: dict, dataset: dict) -> tuple[list[tuple[int, float]], str]:
        m = dataset["test"].shape[0]
        conn.execute("SET jit=false")

        # Debug mode: use single repeated query to diagnose latency degradation
        if self.debug_single_query:
            print(f"Running DEBUG single-query benchmark ({m} iterations of same query)")
            single_query = dataset["test"][0]
            single_answer = dataset["answer"][0][:top]
            if hasattr(single_answer, "tolist"):
                single_answer = single_answer.tolist()
        else:
            print(f"Running sequential benchmark with {m} queries")

        # Pre-convert answers if numpy
        answers_list = dataset["answer"]
        if hasattr(answers_list, "tolist"):
            answers_list = [a[:top].tolist() if hasattr(a, "tolist") else a[:top] for a in answers_list]

        # warmup_query is the single source of truth for the per-benchmark
        # query template + bind shape. Suites with multi-stage queries (e.g.
        # IVFFlat-BQ + rerank) plug them in by overriding warmup_query;
        # measurement here uses the same SQL the warmup loop uses.
        query_sql, bind_fn = self.warmup_query(
            table_name, dataset, metric_ops, top, benchmark
        )

        results = []
        latencies = []
        total_hits = 0
        total_time = 0.0

        # Reuse single cursor to avoid creation overhead
        cursor = conn.cursor()

        pbar = tqdm(range(m), total=m, ncols=80,
                    bar_format="{desc} {n}/{total}: {percentage:3.0f}%|{bar}|")
        for i in pbar:
            # Get query
            query = single_query if self.debug_single_query else dataset["test"][i]

            start = time.perf_counter()
            cursor.execute(query_sql, bind_fn(query))
            result = cursor.fetchall()
            end = time.perf_counter()

            query_time = end - start
            latencies.append(query_time)
            total_time += query_time

            if self.debug_single_query:
                answers = single_answer
            else:
                answers = answers_list[i] if isinstance(answers_list, list) else answers_list[i][:top]
                if hasattr(answers, "tolist"):
                    answers = answers.tolist()

            # Simple set intersection for Recall
            hit = len({p[0] for p in result[:top]} & set(answers))
            total_hits += hit
            results.append((hit, query_time))

            # Update stats every 50 iterations to reduce overhead
            if (i + 1) % 50 == 0 or i == m - 1:
                curr_recall = total_hits / (top * (i + 1))
                curr_qps = (i + 1) / total_time
                curr_p50 = np.percentile(latencies, 50) * 1000
                recall_color = "\033[92m" if curr_recall >= 0.95 else "\033[91m"
                pbar.set_description(f"recall: {recall_color}{curr_recall:.4f}\033[0m QPS: {curr_qps:.2f} P50: {curr_p50:.2f}ms")

        cursor.close()
        pbar.close()
        return results, metric_ops

    def parallel_bench(self, name, table_name, dataset, metric, top, query_clients, benchmark):
        test = dataset["test"]
        answer = dataset["answer"]
        m = test.shape[0]
        total_queries = m * query_clients

        # Single-client probe picks N, then every worker runs N warmup
        # queries. Probe latencies and results stay isolated from the
        # measurement set returned below.
        warmup_n = 0
        if self.warmup != "off" and not self.debug_single_query:
            probe_metric_ops = self._get_metric_operator(metric)
            probe_conn = self.create_connection()
            try:
                self.apply_session_guc(probe_conn, benchmark)
                warmup_n, _converged = self.warmup_for_benchmark(
                    probe_conn, table_name, dataset, probe_metric_ops, top, name,
                    benchmark=benchmark,
                )
            finally:
                probe_conn.close()

        print(f"Running parallel benchmark with {query_clients} clients × {m} queries = {total_queries:,} total"
              + (f" (warmup_n={warmup_n}/worker)" if warmup_n else ""))

        batches = []
        for _ in range(query_clients):
            batch = self.make_batch_args(
                test, answer, top, metric, table_name, benchmark,
                warmup_n=warmup_n,
            )
            batches.append(batch)

        all_results = []
        pbar = tqdm(total=total_queries, ncols=80,
                    bar_format="{desc} {n}/{total}: {percentage:3.0f}%|{bar}|")

        with mp.Pool(processes=query_clients) as pool:
            for batch_result in pool.imap_unordered(self.__class__.process_batch, batches):
                all_results.extend(batch_result)

                # Update progress and stats
                completed = len(all_results)
                pbar.n = completed
                pbar.refresh()

                # Calculate current metrics
                if completed > 0:
                    hits = sum(r[0] for r in all_results)
                    recall = hits / (top * completed)
                    total_time = calculate_coverage([r[1] for r in all_results])
                    qps = completed / total_time
                    latencies = [(r[1][1] - r[1][0]) for r in all_results]
                    p50 = np.percentile(latencies, 50) * 1000

                    recall_color = "\033[92m" if recall >= 0.95 else "\033[91m"
                    pbar.set_description(f"recall: {recall_color}{recall:.4f}\033[0m QPS: {qps:.2f} P50: {p50:.2f}ms")

        pbar.close()
        return all_results, self._get_metric_operator(metric)

    def run_benchmark(self, suite_name: str, name: str, table_name: str, result_dir: str, benchmark: dict,
                      dataset: dict, query_clients):
        top = self.config[suite_name]["top"]
        metric = self.config[suite_name]["metric"]
        m = dataset["test"].shape[0]

        if self.debug:
            print(f"\n🔍 Benchmark Configuration:")
            print(f"    • Name:            {name}")
            print(f"    • Queries:         {m:,}")
            print(f"    • Top-K:           {top}")
            print(f"    • Metric:          {metric}")
            print(f"    • Clients:         {query_clients}")
            print(f"    • Mode:            {'parallel' if query_clients > 1 else 'sequential'}")
            print()

        if query_clients > 1:
            results, metric_ops = self.parallel_bench(suite_name, table_name, dataset, metric, top, query_clients,
                                                      benchmark)
        else:
            conn = self.create_connection()
            results, metric_ops = self.sequential_bench(suite_name, table_name, conn, metric, top, benchmark, dataset)
            conn.close()

        self.results[suite_name]["metric_ops"] = metric_ops

        recall, qps, p50, p99 = calculate_metrics(results, top, m, query_clients)
        print(f"Top: {top} | Recall: {recall:.4f} | QPS: {qps:.2f} | P50: {p50:.2f}ms | P99: {p99:.2f}ms")

        self.results[suite_name][name] = {
            "recall": recall,
            "qps": qps,
            "p50_latency": p50,
            "p99_latency": p99,
        }

    def print_summary_table(self, suite_name: str):
        """Print a summary table of all benchmark results for a suite."""
        benchmarks = self.config[suite_name].get("benchmarks", {})
        results = self.results.get(suite_name, {})

        if not benchmarks:
            return

        # Determine columns based on benchmark parameters
        first_bench = next(iter(benchmarks.values()))
        has_ef_search = "efSearch" in first_bench
        has_nprob = "nprob" in first_bench

        # Build header and rows
        if has_ef_search:
            header = "| EF Search | Recall | QPS    | P50 (ms) | P99 (ms) |"
            sep =    "|-----------|--------|--------|----------|----------|"
        elif has_nprob:
            header = "| Probes    | Epsilon | Recall | QPS    | P50 (ms) | P99 (ms) |"
            sep =    "|-----------|---------|--------|--------|----------|----------|"
        else:
            return

        sb = results.get("shared_buffers", "N/A")
        idx_size = results.get("index_size", "N/A")
        qc = results.get("query_clients", 1)

        print(f"\n{'=' * len(sep)}")
        print(f"  Results Summary: {suite_name}")
        print(f"  shared_buffers: {sb} | clients: {qc} | index_size: {idx_size}")
        print(f"{'=' * len(sep)}")
        print(header)
        print(sep)

        for name, benchmark in benchmarks.items():
            r = results.get(name, {})
            if "recall" not in r:
                continue

            if has_ef_search:
                print(f"| {benchmark['efSearch']:<9} "
                      f"| {r['recall']:.4f} "
                      f"| {r['qps']:>6.2f} "
                      f"| {r['p50_latency']:>8.2f} "
                      f"| {r['p99_latency']:>8.2f} |")
            elif has_nprob:
                print(f"| {benchmark['nprob']:<9} "
                      f"| {benchmark['epsilon']:<7} "
                      f"| {r['recall']:.4f} "
                      f"| {r['qps']:>6.2f} "
                      f"| {r['p50_latency']:>8.2f} "
                      f"| {r['p99_latency']:>8.2f} |")

        print()

    def run_benchmarks(self, suite_name: str, table_name: str, dataset: dict, query_clients):
        # Limit queries if requested
        if self.max_queries and dataset["test"].shape[0] > self.max_queries:
            dataset = {**dataset,
                       "test": dataset["test"][:self.max_queries],
                       "answer": dataset["answer"][:self.max_queries]}

        if self.warmup == "off":
            print("[warmup] mode=off: skipping pg_prewarm")
        else:
            self.prewarm_index(table_name)

        benchmarks = self.config[suite_name]["benchmarks"]
        first_benchmark = next(iter(benchmarks.values()), None)
        if first_benchmark is not None:
            metric_ops = self._get_metric_operator(dataset.get("metric", "cos"))
            top = self.config[suite_name].get("top", 10)
            self.verify_index_usage(table_name, dataset, metric_ops, top, first_benchmark)

        for name, benchmark in benchmarks.items():
            print(f"Running benchmark: {benchmark}")
            result_dir = f"./results/{suite_name}/"
            os.makedirs(result_dir, exist_ok=True)
            self.results[suite_name][name] = {}
            self.run_benchmark(suite_name, name, table_name, result_dir, benchmark, dataset, query_clients)

        self.print_summary_table(suite_name)

    def generate_markdown_result(self) -> None:
        raise NotImplementedError("generate_markdown_result method should be implemented in subclasses.")

    def run_suite(self, name: str):
        os.makedirs(f"./results/{name}", exist_ok=True)
        self.results[name] = {}
        self.results[name]["query_clients"] = self.query_clients
        if self._system_report_content:
            self.results[name]["system_report"] = self._system_report_content

        dataset_name = self.config[name]["dataset"]
        table_name = dataset_name.replace("-", "_")
        config = self.config[name]
        if self.debug:
            print(f"\n⚙️  Suite Configuration:")
            print(f"    • Test Name:       {name}")
            print(f"    • Table:           {table_name}")
            print(f"    • Dataset:         {dataset_name}")
            print(f"    • PG Parallel Workers: {config.get('pg_parallel_workers')}")
            print(f"    • Metric:          {config.get('metric')}")
            print(f"    • Centroids Table: {self.centroids_table or 'None'}")
            print(f"    • Benchmarks to run: {len(config.get('benchmarks', {}))}")
            print()

        # Initialize system monitor only for local databases
        if self.is_local_db:
            self.system_monitor = SystemMonitor(
                results_dir=f"./results/{name}",
                devices=self.devices if self.devices else None,
                sample_interval=1.0  # Sample every second
            )
            self.system_monitor.start()  # Start background monitoring
        else:
            self.system_monitor = None

        self.init_ext(name)

        # Capture PostgreSQL settings for this run
        conn = self.create_connection()
        try:
            rows = conn.execute(
                "SELECT name, setting, unit FROM pg_settings "
                "WHERE name IN ('shared_buffers', 'maintenance_work_mem')"
            ).fetchall()
            for setting_name, setting_val, unit in rows:
                val = int(setting_val)
                if unit == '8kB':
                    size_mb = val * 8 / 1024
                    self.results[name][setting_name] = f"{size_mb:.0f}MB" if size_mb < 1024 else f"{size_mb / 1024:.0f}GB"
                elif unit == 'kB':
                    size_mb = val / 1024
                    self.results[name][setting_name] = f"{size_mb:.0f}MB" if size_mb < 1024 else f"{size_mb / 1024:.0f}GB"
                else:
                    self.results[name][setting_name] = f"{setting_val}{unit}" if unit else setting_val
        except psycopg.Error:
            pass

        # Capture non-default settings and write full pg_settings.csv
        try:
            import csv as _csv
            all_settings = conn.execute(
                "SELECT name, setting, unit, source FROM pg_settings ORDER BY name"
            ).fetchall()
            settings_dir = f"./results/{name}"
            os.makedirs(settings_dir, exist_ok=True)
            with open(f"{settings_dir}/pg_settings.csv", "w", newline="") as _f:
                w = _csv.writer(_f)
                w.writerow(["name", "setting", "unit", "source"])
                for row in all_settings:
                    w.writerow(row)
            nondefault = [
                [r[0], f"{r[1]}{r[2] or ''}", r[3]]
                for r in all_settings
                if r[3] not in ("default", "client")
            ]
            if nondefault:
                self.results[name]["pg_settings_nondefault"] = nondefault
        except Exception:
            pass

        # Initialize PG stats collector
        self.pg_stats_collector = PGStatsCollector(conn)

        # Only capture baseline if table already exists (skip_add_embeddings case)
        if self.skip_add_embeddings:
            self.pg_stats_collector.capture_snapshot("baseline", table_name)

        # 1. LOAD DATASET (Unified)
        selectivity = self.config[name].get("selectivity")
        ds = datasets.get_dataset(dataset_name, selectivity=selectivity)
        # Compatibility mapping
        ds["answer"] = ds.pop("neighbors")

        # 2. ADD EMBEDDINGS
        if not self.skip_add_embeddings:
            if self.system_monitor:
                self.system_monitor.mark_phase("load_start")
            pg_parallel_workers = self.config[name].get("pg_parallel_workers")
            self.add_embeddings(name, table_name, ds, pg_parallel_workers=pg_parallel_workers)
            if hasattr(ds["train"], "close"):
                ds["train"].close()

            # Free memory
            del ds["train"]
            gc.collect()

            if self.system_monitor:
                self.system_monitor.mark_phase("load_end")
            self.pg_stats_collector.capture_snapshot("after_load", table_name)

        if self.centroids is not None and not self.skip_index_creation:
            self.add_centroids_to_table(self.centroids)

        if "metric" in self.config[name]:
            ds["metric"] = self.config[name]["metric"]

        if not self.skip_index_creation:
            if self.system_monitor:
                self.system_monitor.mark_phase("index_start")
            self.create_index(name, table_name, ds)
            if self.system_monitor:
                self.system_monitor.mark_phase("index_end")
            self.pg_stats_collector.capture_snapshot("after_index", table_name)
        else:
            print("Skipping index creation as requested.")

        self.calculate_index_size(name, table_name)

        if not self.build_only:
            if self.system_monitor:
                self.system_monitor.mark_phase("benchmark_start")
            self.run_benchmarks(name, table_name, ds, self.query_clients)
            if self.system_monitor:
                self.system_monitor.mark_phase("benchmark_end")
            self.pg_stats_collector.capture_snapshot("after_benchmark", table_name)
        else:
            print("Build-only mode: skipping benchmarks.")

        if self.system_monitor:
            self.system_monitor.stop()

        conn.close()

    def get_monitoring_data(self, suite_name: str) -> tuple:
        """
        Get formatted monitoring data for reports.

        Returns:
            Tuple of (system_metrics_md, pg_stats_md, dashboard_path)
        """
        system_metrics_md = None
        pg_stats_md = None
        dashboard_path = None

        if self.system_monitor:
            system_metrics_md = self.system_monitor.format_for_report()
            dashboard_path = self.system_monitor.generate_dashboard(suite_name)
            self.system_monitor.save_csv(f"{suite_name}_system_metrics.csv")

        if self.pg_stats_collector:
            pg_stats_md = self.pg_stats_collector.format_for_report()

        return system_metrics_md, pg_stats_md, dashboard_path

    def run(self):
        os.makedirs("./results", exist_ok=True)

        # Auto-detect IO device if not specified and database is local
        if self.is_local_db and self.devices is None:
            conn = self.create_connection()
            detected_devices = detect_pg_io_device(conn)
            conn.close()

            if detected_devices:
                self.devices = detected_devices
                self.debug_log(f"Auto-detected PostgreSQL data device: {', '.join(self.devices)}")
            else:
                print("Warning: Could not auto-detect PostgreSQL data device. IO monitoring disabled.")
                self.devices = []

        # Generate system report only for local databases
        self._system_report_content = None
        if self.is_local_db:
            self._system_report_content = generate_system_report()

        for suite_name in self.config:
            mode = self.config[suite_name].get("mode", "readonly")
            print(f"Running test: {suite_name} (mode={mode})")
            if mode == "workload":
                from workload import run_workload
                run_workload(suite_name, self.config[suite_name], self.url,
                             chunk_size=self.chunk_size,
                             num_threads=self.max_load_threads,
                             suite=self)
            elif mode == "filtered":
                from filtered import run_filtered
                run_filtered(suite_name, self.config[suite_name], self.url,
                             chunk_size=self.chunk_size,
                             num_threads=self.max_load_threads)
            else:
                self.run_suite(suite_name)
        self.generate_markdown_result()
