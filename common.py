import argparse
import gc
import multiprocessing as mp
import os
import threading
import time

import numpy as np
import psutil
import psycopg
import yaml
from tqdm import tqdm

import datasets
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
                 debug_single_query: bool = False):
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
            dbname="postgres",
            autocommit=True,
            **self.keepalive_kwargs,
        )
        return conn

    def make_batch_args(self, test, answer, top, metric_ops, table_name, benchmark):
        return test, answer, top, metric_ops, table_name

    def prewarm_index(self, table_name: str):
        raise NotImplementedError("prewarm_index should be implemented in subclasses.")

    def add_embeddings_from_hdf5(self, suite_name, name, train, pg_parallel_workers):
        """Parallel add embeddings to the database from HDF5."""
        n, dim = train.shape
        start_time = time.perf_counter()

        conn = self.create_connection()
        if self.debug:
            print(f"\n📦 Load Configuration (HDF5):")
            print(f"    • Table:           {name}")
            print(f"    • Rows:            {n:,}")
            print(f"    • Dimensions:      {dim}")
            print(f"    • Load Threads:    {self.max_load_threads}")
            print(f"    • Chunk Size:      {self.chunk_size:,}")
            print(f"    • Overwrite:       {self.overwrite_table}")
            print()

        if self.overwrite_table:
            print(f"Dropping existing table {name}...", end="", flush=True)
            conn.execute(f"DROP TABLE IF EXISTS {name}")
            print(" done!")

        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM {name} LIMIT 1")
            table_exists = True
        except psycopg.errors.UndefinedTable:
            table_exists = False

        if table_exists:
            print(f"Table {name} already exists, using it")
            return

        conn.execute(f"CREATE TABLE {name} (id integer, embedding vector({dim}))")
        conn.execute(f"ALTER TABLE {name} SET (parallel_workers = {pg_parallel_workers})")
        conn.close()

        max_load_threads = self.max_load_threads

        def load_chunk(chunk_start, chunk_len):
            data = train[chunk_start: chunk_start + chunk_len]
            conn = self.create_connection()
            with conn.cursor().copy(
                    f"COPY {name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["integer", "vector"])
                for i, vec in enumerate(data):
                    copy.write_row((chunk_start + i, vec))
                while conn.pgconn.flush() == 1:
                    time.sleep(0)
            conn.close()

        chunk_size = self.chunk_size
        pbar = tqdm(desc="Adding embeddings", total=n, ncols=80, unit_scale=True, unit="rows")
        threads = []
        for i in range(0, n, chunk_size):
            chunk_start = i
            chunk_len = min(chunk_size, n - i)
            t = threading.Thread(
                target=load_chunk, args=(chunk_start, chunk_len))
            threads.append(t)

            if len(threads) == max_load_threads or chunk_start + chunk_len >= n:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                    pbar.update(chunk_size)
                threads = []
        pbar.close()

        end_time = time.perf_counter()
        self.results[suite_name]["load_time"] = int(round(end_time - start_time))
        print(f'Dataset load time: {self.results[suite_name]["load_time"]}s')

    def add_embeddings_from_npy(self, suite_name, name, ds: dict):
        """
        Add embeddings from NPY files to the database.
        - Deep1B (mmap) -> Uses Optimized Chunk Loading.
        - LAION (generator) -> Uses Standard Sequential Loading.
        """
        dim = ds["dim"]
        n = ds["num"]
        data = ds["train"]

        is_sliceable = hasattr(data, "__getitem__") and hasattr(data, "shape")
        conn = self.create_connection()

        # --- Table Setup ---
        if self.overwrite_table:
            print(f"Dropping existing table {name}...", end="", flush=True)
            conn.execute(f"DROP TABLE IF EXISTS {name}")
            print(" done!")

        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM {name} LIMIT 1")
            table_exists = True
        except psycopg.errors.UndefinedTable:
            table_exists = False

        if table_exists:
            print(f"Table {name} already exists, using it")
            conn.close()
            return

        print(f"Creating TABLE {name}...")
        conn.execute(f"CREATE TABLE {name} (id integer, embedding vector({dim}))")
        conn.commit()
        conn.close()

        # --- Loading Data ---
        start_time = time.perf_counter()

        # Default chunk size 100k for smoothness
        chunk_size = self.chunk_size if self.chunk_size else 100_000

        if is_sliceable:
            # === OPTIMIZED PATH (Deep1B) ===
            num_threads = self.max_load_threads or 1

            def load_chunk(chunk_start, chunk_len):
                t_conn = self.create_connection()
                chunk_data = data[chunk_start: chunk_start + chunk_len]

                # Cast if needed
                if chunk_data.dtype != np.float32:
                    chunk_data = chunk_data.astype(np.float32)

                with t_conn.cursor().copy(
                        f"COPY {name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
                ) as copy:
                    copy.set_types(["integer", "vector"])
                    for i, vec in enumerate(chunk_data):
                        copy.write_row((chunk_start + i, vec))

                    while t_conn.pgconn.flush() == 1:
                        time.sleep(0)
                t_conn.close()

            # Initialize Progress Bar in "rows"
            pbar = tqdm(desc="Adding embeddings", total=n, unit="rows", ncols=80, unit_scale=True)

            if num_threads > 1: # Careful with python GIL and I/O bound
                print(f"Parallel load enabled for {name}. Threads: {num_threads}")
                threads = []
                batch_rows = 0
                for i in range(0, n, chunk_size):
                    chunk_len = min(chunk_size, n - i)
                    t = threading.Thread(target=load_chunk, args=(i, chunk_len))
                    threads.append(t)
                    batch_rows += chunk_len

                    if len(threads) >= num_threads or (i + chunk_len) >= n:
                        for thread in threads:
                            thread.start()
                        for thread in threads:
                            thread.join()
                        pbar.update(batch_rows)
                        threads = []
                        batch_rows = 0
            else:
                print(f"Sequential load enabled for {name} (Optimized Mmap)")
                for i in range(0, n, chunk_size):
                    chunk_len = min(chunk_size, n - i)
                    load_chunk(i, chunk_len)
                    pbar.update(chunk_len)

            pbar.close()

        else:
            # === GENERATOR PATH (LAION) ===
            print(f"Sequential load for {name} (Generator source)")
            conn = self.create_connection()
            # Initialize Pbar for generator
            pbar = tqdm(desc="Adding embeddings", total=n, unit="rows", ncols=80, unit_scale=True)

            with conn.cursor().copy(
                    f"COPY {name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["integer", "vector"])
                for i, vec in data:
                    copy.write_row((i, vec))
                    while conn.pgconn.flush() == 1:
                        time.sleep(0)
                    pbar.update(1)
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

    def monitor_index_build(self, event: threading.Event):
        conn = self.create_connection()
        with conn.cursor() as acur:
            blocks_total = 0
            phase = "initializing"

            # Wait briefly for progress info, but don't block forever
            for _ in range(10):  # Try for 5 seconds
                if event.is_set():
                    conn.close()
                    return
                acur.execute("SELECT blocks_total, phase FROM pg_stat_progress_create_index")
                result = acur.fetchone()
                if result:
                    blocks_total = result[0] or 0
                    phase = result[1] or "initializing"
                    if blocks_total > 0:
                        break
                time.sleep(0.5)

            # Show progress bar even if blocks_total is 0 (e.g., during clustering)
            pbar = tqdm(smoothing=0.0, total=max(blocks_total, 1), desc=f"Building index ({phase})", ncols=100,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]")

            while True:
                if event.is_set():
                    pbar.update(pbar.total - pbar.n)
                    pbar.close()
                    conn.close()
                    break
                acur.execute("SELECT blocks_done, blocks_total, phase FROM pg_stat_progress_create_index")
                result = acur.fetchone()
                if result:
                    blocks_done = result[0] or 0
                    new_total = result[1] or 0
                    new_phase = result[2] or phase

                    # Update phase description
                    if new_phase != phase:
                        phase = new_phase
                        pbar.set_description(f"Building index ({phase})")

                    # Update total if it changed and is non-zero
                    if new_total > 0 and new_total != pbar.total:
                        pbar.total = new_total
                        pbar.n = 0
                        pbar.refresh()

                    if new_total > 0:
                        pbar.update(max(blocks_done - pbar.n, 0))
                time.sleep(0.5)

    def create_index(self, suite_name: str, table_name: str, dataset: dict) -> tuple[
        threading.Event, threading.Thread]:
        os.makedirs(f"./results/{suite_name}/index_build", exist_ok=True)
        conn = self.create_connection()
        print(f"Dropping index {table_name}_embedding_idx...", end="", flush=True)
        try:
            conn.execute(f"DROP INDEX IF EXISTS {table_name}_embedding_idx")
            print(" done!")
        except Exception:
            print(" failed!")
        finally:
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
        with conn.cursor() as acur:
            acur.execute(
                f'''
                SELECT pg_size_pretty(pg_relation_size(indexrelid))
                FROM pg_stat_all_indexes i JOIN pg_class c ON i.relid=c.oid
                WHERE i.relname='{table_name}';
                '''
            )
            result = acur.fetchone()
            self.results[suite_name]["index_size"] = result[0]
            print(f"Index size: {result[0]}")
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

        # Build query template (psycopg3 handles prepared statement caching)
        query_sql = f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s LIMIT {top}"

        results = []
        latencies = []  # Pre-allocate for efficiency

        # Reuse single cursor to avoid creation overhead
        cursor = conn.cursor()

        pbar = tqdm(range(m), total=m, ncols=80,
                    bar_format="{desc} {n}/{total}: {percentage:3.0f}%|{bar}|")
        for i in pbar:
            # Get query
            query = single_query if self.debug_single_query else dataset["test"][i]

            start = time.perf_counter()
            cursor.execute(query_sql, (query,))
            result = cursor.fetchall()
            end = time.perf_counter()

            query_time = end - start
            latencies.append(query_time)

            if self.debug_single_query:
                answers = single_answer
            else:
                answers = answers_list[i] if isinstance(answers_list, list) else answers_list[i][:top]
                if hasattr(answers, "tolist"):
                    answers = answers.tolist()

            # Simple set intersection for Recall
            hit = len({p[0] for p in result[:top]} & set(answers))
            results.append((hit, query_time))

            # Update stats every 50 iterations to reduce overhead
            if (i + 1) % 50 == 0 or i == m - 1:
                curr_recall = sum(r[0] for r in results) / (top * (i + 1))
                total_time = sum(latencies)
                curr_qps = (i + 1) / total_time
                curr_p50 = np.percentile(latencies, 50) * 1000
                recall_color = "\033[92m" if curr_recall >= 0.95 else "\033[91m"
                pbar.set_description(f"recall: {recall_color}{curr_recall:.4f}\033[0m QPS: {curr_qps:.2f} P50: {curr_p50:.2f}ms")

        cursor.close()
        pbar.close()
        return results, metric_ops

    def parallel_bench(self, name, table_name, dataset, metric_ops, top, query_clients, benchmark):
        test = dataset["test"]
        answer = dataset["answer"]
        m = test.shape[0]
        total_queries = m * query_clients

        print(f"Running parallel benchmark with {query_clients} clients × {m} queries = {total_queries:,} total")

        batches = []
        for _ in range(query_clients):
            batch = self.make_batch_args(test, answer, top, metric_ops, table_name, benchmark)
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
        return all_results, metric_ops

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

    def run_benchmarks(self, suite_name: str, table_name: str, dataset: dict, query_clients):
        # Prewarm index once before all benchmarks
        self.prewarm_index(table_name)

        for name, benchmark in self.config[suite_name]["benchmarks"].items():
            print(f"Running benchmark: {benchmark}")
            result_dir = f"./results/{suite_name}/"
            os.makedirs(result_dir, exist_ok=True)
            self.results[suite_name][name] = {}
            self.run_benchmark(suite_name, name, table_name, result_dir, benchmark, dataset, query_clients)

    def generate_markdown_result(self):
        return NotImplementedError("generate_markdown_result method should be implemented in subclasses.")

    def run_suite(self, name: str):
        os.makedirs(f"./results/{name}", exist_ok=True)
        self.results[name] = {}

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

        # Initialize PG stats collector
        conn = self.create_connection()
        self.pg_stats_collector = PGStatsCollector(conn)

        # Only capture baseline if table already exists (skip_add_embeddings case)
        if self.skip_add_embeddings:
            self.pg_stats_collector.capture_snapshot("baseline", table_name)

        # 1. LOAD DATASET (Unified)
        ds = datasets.get_dataset(dataset_name)
        # Compatibility mapping
        ds["answer"] = ds.pop("neighbors")

        # 2. ADD EMBEDDINGS
        if not self.skip_add_embeddings:
            if self.system_monitor:
                self.system_monitor.mark_phase("load_start")
            ds_type = ds["type"]
            if ds_type == "hdf5":
                self.add_embeddings_from_hdf5(name, table_name, ds["train"], self.config[name]["pg_parallel_workers"])
                if hasattr(ds["train"], "close"):
                    ds["train"].close()
            elif ds_type in ["laion-multipart", "deep1b-mmap"]:
                self.add_embeddings_from_npy(name, table_name, ds)

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
            self.calculate_index_size(name, table_name)
            if self.system_monitor:
                self.system_monitor.mark_phase("index_end")
            self.pg_stats_collector.capture_snapshot("after_index", table_name)
        else:
            print("Skipping index creation as requested.")

        if self.system_monitor:
            self.system_monitor.mark_phase("benchmark_start")
        self.run_benchmarks(name, table_name, ds, self.query_clients)
        if self.system_monitor:
            self.system_monitor.mark_phase("benchmark_end")
            self.system_monitor.stop()  # Stop background monitoring
        self.pg_stats_collector.capture_snapshot("after_benchmark", table_name)

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
        if self.is_local_db:
            generate_system_report("./results")

        for suite_name in self.config:
            print(f"Running test: {suite_name}")
            self.run_suite(suite_name)
        self.generate_markdown_result()
