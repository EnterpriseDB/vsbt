"""
pgvector Benchmark Suite

Benchmarks vector search using the pgvector extension with HNSW or
IVFFlat indexes (vanilla and BQ + rerank variants) for PostgreSQL.

A single TestSuite handles all three index types. The per-index
variation lives in INDEX_SPECS — small functions that build the CREATE
INDEX statement, the per-benchmark session GUCs, and the per-benchmark
query template. Everything else (warmup, sequential / parallel
benchmark, monitoring, report generation) is shared.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import psycopg
import pgvector.psycopg

import common
from results import ResultsManager

# Default candidate-set amplification for IVFFlat-BQ + rerank: a query
# pulls top * DEFAULT_RERANK_AMP rows from the binary-quantized index
# before re-sorting by exact distance.
DEFAULT_RERANK_AMP = 20

# Operator + opclass per metric, shared across all pgvector index types.
_METRIC_OPS = {
    "l2": "<->", "euclidean": "<->",
    "cos": "<=>", "angular": "<=>",
    "dot": "<#>", "ip": "<#>",
}
_METRIC_FUNCS = {
    "l2": "vector_l2_ops", "euclidean": "vector_l2_ops",
    "cos": "vector_cosine_ops",
    "ip": "vector_ip_ops", "dot": "vector_ip_ops",
}
# Hamming distance on a sign-bit binary quantization preserves the
# cosine/angular ordering of the original vectors, but not the L2 or raw
# inner-product ordering — so BQ + rerank only makes sense for cos/angular.
_BQ_RERANK_METRICS = {"cos", "angular"}


def _metric_op(metric: str) -> str:
    if metric not in _METRIC_OPS:
        raise ValueError(f"Unsupported metric type: {metric}")
    return _METRIC_OPS[metric]


def _metric_func(metric: str) -> str:
    if metric not in _METRIC_FUNCS:
        raise ValueError(f"Unsupported metric type: {metric}")
    return _METRIC_FUNCS[metric]


@dataclass(frozen=True)
class IndexSpec:
    """Per-index-type variation. A TestSuite holds one of these."""

    index_type: str
    # Passed through as `suite_type` to ResultsManager so the
    # markdown/csv layer renders the right columns. HNSW uses upstream's
    # "pgvector"; new index types use distinct values that route through
    # the additive branches in results.py.
    suite_type: str

    # Builds the per-benchmark query: returns (sql_template, bind_fn).
    # Used by both the warmup loop and sequential_bench measurement loop.
    query_template: Callable[[str, dict, str, int, dict], tuple[str, Callable]]

    # "single" -> bind is `(query,)`; "two_stage" -> `(query, rerank_limit, query)`.
    # Held separately because process_batch runs in a multiprocessing worker
    # where lambdas can't cross the pickle boundary.
    bind_kind: str

    # Per-benchmark session GUCs (`SET ...` statements as full SQL strings).
    session_gucs: Callable[[dict], list[str]]

    # `CREATE INDEX ...` statement.
    create_index_sql: Callable[[str, dict, dict], str]

    # Optional debug print invoked from create_index.
    debug_print: Callable[[dict, dict], None]

    # (benchmark-dict key, column header) pairs for the new generic
    # summary path. Empty for HNSW so it falls through to upstream's
    # legacy efSearch branch in print_summary_table and results.py.
    bench_param_columns: tuple[tuple[str, str], ...]

    # (label, extractor(config, results) -> str) rows for the per-run
    # markdown "Configuration" table. Empty for HNSW (handled by the
    # legacy "pgvector" branch in results.py).
    config_columns: tuple


def _ivfflat_resolve_lists(config: dict, dataset: dict) -> int:
    lists = config["lists"]
    if isinstance(lists, str) and lists.strip().lower() == "auto":
        return max(1, int(math.sqrt(dataset["num"])))
    if not isinstance(lists, int) or lists < 1:
        raise ValueError(
            f"lists must be a positive integer or 'auto'; got {lists!r}"
        )
    return lists


def _resolve_rerank_amp(benchmark: dict) -> int:
    rerank_amp = benchmark.get("rerank_limit_amplify_factor", DEFAULT_RERANK_AMP)
    if not isinstance(rerank_amp, int) or rerank_amp < 1:
        raise ValueError(
            f"rerank_limit_amplify_factor must be a positive integer; "
            f"got {rerank_amp!r}"
        )
    return rerank_amp


def _hnsw_query_template(table_name, _dataset, metric_ops, top, _benchmark):
    sql_text = (
        f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s "
        f"LIMIT {top}"
    )
    return sql_text, (lambda q: (q,))


def _hnsw_session_gucs(benchmark):
    return [
        f"SET hnsw.ef_search={benchmark['efSearch']}",
        "SET enable_seqscan = off",
    ]


def _hnsw_create_index_sql(table_name, config, dataset):
    metric_func = _metric_func(dataset["metric"])
    return (
        f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
        f"USING hnsw (embedding {metric_func}) "
        f"WITH (m = {config['m']}, ef_construction = {config['efConstruction']})"
    )


def _hnsw_debug_print(config, dataset):
    print(f"\n🔧 Index Configuration (HNSW):")
    print(f"    • M:               {config['m']}")
    print(f"    • EF Construction: {config['efConstruction']}")
    print(f"    • Metric Function: {_metric_func(dataset['metric'])}")
    print()


def _ivfflat_query_template(table_name, _dataset, metric_ops, top, _benchmark):
    sql_text = (
        f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s "
        f"LIMIT {top}"
    )
    return sql_text, (lambda q: (q,))


def _ivfflat_session_gucs(benchmark):
    return [
        f"SET ivfflat.probes TO {benchmark['probes']}",
        "SET enable_seqscan = off",
    ]


def _ivfflat_create_index_sql(table_name, config, dataset):
    lists = _ivfflat_resolve_lists(config, dataset)
    metric_func = _metric_func(dataset["metric"])
    return (
        f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
        f"USING ivfflat (embedding {metric_func}) WITH (lists = {lists})"
    )


def _ivfflat_debug_print(config, dataset):
    lists = _ivfflat_resolve_lists(config, dataset)
    print(f"\n🔧 Index Configuration (IVFFlat):")
    print(f"    • Lists:           {lists}")
    print(f"    • Metric Function: {_metric_func(dataset['metric'])}")
    print()


def _bq_rerank_two_stage_sql(table_name, dim, rerank_op, top):
    return (
        f"SELECT id FROM ("
        f"SELECT id, embedding FROM {table_name} "
        f"ORDER BY binary_quantize(embedding)::bit({dim}) <~> "
        f"binary_quantize(%s::vector({dim}))::bit({dim}) "
        f"LIMIT %s::int"
        f") sub "
        f"ORDER BY embedding {rerank_op} %s::vector({dim}) "
        f"LIMIT {top}"
    )


def _bq_rerank_query_template(table_name, dataset, metric_ops, top, benchmark):
    dim = dataset["dim"]
    rerank_limit = top * _resolve_rerank_amp(benchmark)
    sql_text = _bq_rerank_two_stage_sql(table_name, dim, metric_ops, top)
    return sql_text, (lambda q: (q, rerank_limit, q))


def _bq_rerank_session_gucs(benchmark):
    return _ivfflat_session_gucs(benchmark)


def _bq_rerank_create_index_sql(table_name, config, dataset):
    metric = dataset["metric"]
    if metric not in _BQ_RERANK_METRICS:
        raise ValueError(
            f"ivfflat_bq_rerank only supports cosine/angular metrics "
            f"(Hamming on binary_quantize preserves only cosine ordering); "
            f"got metric={metric!r}"
        )
    lists = _ivfflat_resolve_lists(config, dataset)
    dim = dataset["dim"]
    return (
        f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
        f"USING ivfflat ((binary_quantize(embedding)::bit({dim})) bit_hamming_ops) "
        f"WITH (lists = {lists})"
    )


def _bq_rerank_debug_print(config, dataset):
    lists = _ivfflat_resolve_lists(config, dataset)
    print(f"\n🔧 Index Configuration (IVFFlat BQ Rerank):")
    print(f"    • Lists:           {lists}")
    print(f"    • Dimensions:      {dataset['dim']}")
    print()


_IVFFLAT_CONFIG_COLUMNS = (
    ("Lists", lambda c, r: str(c.get("lists", r.get("lists", "N/A")))),
)


INDEX_SPECS = {
    "hnsw": IndexSpec(
        index_type="hnsw",
        suite_type="pgvector",
        query_template=_hnsw_query_template,
        bind_kind="single",
        session_gucs=_hnsw_session_gucs,
        create_index_sql=_hnsw_create_index_sql,
        debug_print=_hnsw_debug_print,
        # Empty: HNSW falls through to upstream's legacy efSearch branch
        # in print_summary_table and the "pgvector" branch in results.py.
        bench_param_columns=(),
        config_columns=(),
    ),
    "ivfflat": IndexSpec(
        index_type="ivfflat",
        suite_type="pgvector-ivfflat",
        query_template=_ivfflat_query_template,
        bind_kind="single",
        session_gucs=_ivfflat_session_gucs,
        create_index_sql=_ivfflat_create_index_sql,
        debug_print=_ivfflat_debug_print,
        bench_param_columns=(("probes", "Probes"),),
        config_columns=_IVFFLAT_CONFIG_COLUMNS,
    ),
    "ivfflat_bq_rerank": IndexSpec(
        index_type="ivfflat_bq_rerank",
        suite_type="pgvector-ivfflat-bq-rerank",
        query_template=_bq_rerank_query_template,
        bind_kind="two_stage",
        session_gucs=_bq_rerank_session_gucs,
        create_index_sql=_bq_rerank_create_index_sql,
        debug_print=_bq_rerank_debug_print,
        bench_param_columns=(
            ("probes", "Probes"),
            ("rerank_limit_amplify_factor", "Rerank Amp"),
        ),
        config_columns=_IVFFLAT_CONFIG_COLUMNS,
    ),
}


# HNSW build memory / on-disk size estimators.
def _maxalign(x):
    return (x + 7) & ~7


def _estimate_hnsw_graph_memory(num_vectors: int, dim: int, m: int) -> int:
    """Estimate maintenance_work_mem needed for an in-memory HNSW build.

    Based on pgvector's in-memory graph layout (HnswElementData, neighbor
    arrays, and vector storage). Each node at level L consumes:

      MAXALIGN(sizeof(HnswElementData))        ~128 bytes
      MAXALIGN(8 + 4*dim)                      vector value
      MAXALIGN(8 * (L+1))                      neighbor list pointers
      MAXALIGN(8 + 32*m)                       layer 0 neighbor array
      L * MAXALIGN(8 + 16*m)                   upper layer neighbor arrays

    Levels follow P(level >= L) = (1/m)^L, so the expected upper-layer
    overhead per node is (1/(m-1)) * (8 + MAXALIGN(8 + 16*m)).
    """
    element_size = 128
    vector_size = _maxalign(8 + 4 * dim)
    layer0_neighbors = _maxalign(8 + 32 * m)
    layer0_ptrs = _maxalign(8)
    upper_layer_cost = _maxalign(8) + _maxalign(8 + 16 * m)
    upper_layer_fraction = 1.0 / (m - 1) if m > 1 else 0
    avg_per_node = (
        element_size + vector_size + layer0_ptrs + layer0_neighbors
        + upper_layer_fraction * upper_layer_cost
    )
    return int(num_vectors * avg_per_node)


def _estimate_hnsw_index_size(num_vectors: int, dim: int, m: int) -> int:
    """Estimate on-disk HNSW index size based on pgvector's page layout.

    Validated against:
      dim=96,  m=16, 1B vectors  → predicts 632 GB (actual 646 GB, ~2% off)
      dim=768, m=16, 5M vectors  → predicts 19.0 GB (actual 18.8 GB, ~1% off)
    """
    USABLE_PAGE = 8192 - 40
    TUPLE_OVERHEAD = 32
    NEIGHBOR_SIZE = 6

    vector_bytes = _maxalign(8 + 4 * dim)
    neighbor_bytes_l0 = _maxalign(4 + 2 * m * NEIGHBOR_SIZE)
    upper_neighbor_avg = _maxalign(4 + m * NEIGHBOR_SIZE) / (m - 1) if m > 1 else 0
    raw_node_size = TUPLE_OVERHEAD + vector_bytes + neighbor_bytes_l0 + int(upper_neighbor_avg)

    nodes_per_page = max(1, USABLE_PAGE // raw_node_size)
    actual_bytes_per_node = USABLE_PAGE / nodes_per_page
    return int(actual_bytes_per_node * num_vectors)


def build_arg_parse():
    """Build argument parser for pgvector benchmark suite."""
    parser = argparse.ArgumentParser(description="pgvector Benchmark Suite")
    common.build_arg_parse(parser)
    return parser


class TestSuite(common.TestSuite):
    """Single suite for HNSW / IVFFlat / IVFFlat-BQ-Rerank, dispatched by
    the YAML `indexType` field via INDEX_SPECS. The HNSW path produces
    byte-identical SQL/GUCs/reports to upstream."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # All sub-configs in one YAML must agree on indexType — the suite
        # is created once and the spec is fixed.
        index_types = {
            cfg.get("indexType", "hnsw") for cfg in self.config.values()
        }
        if len(index_types) > 1:
            raise ValueError(
                f"Mixed indexTypes in one suite are not supported: {sorted(index_types)}"
            )
        index_type = next(iter(index_types))
        if index_type not in INDEX_SPECS:
            raise ValueError(
                f"Unknown indexType {index_type!r}; "
                f"expected one of {sorted(INDEX_SPECS)}"
            )
        self.spec = INDEX_SPECS[index_type]
        self._batch_dataset = None

    def create_connection(self):
        """Create a database connection with pgvector support."""
        conn = super().create_connection()
        pgvector.psycopg.register_vector(conn)
        return conn

    def init_ext(self, suite_name: Optional[str] = None):
        """Initialize required PostgreSQL extensions."""
        conn = super().create_connection()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        """Prewarm the index into memory for consistent benchmarking."""
        index_name = f"{table_name}_embedding_idx"
        conn = self.create_connection()
        self.check_index_fits_shared_buffers(conn, index_name, table_name)
        print("Prewarming the index into shared_buffers...", end="", flush=True)
        try:
            prewarm_start = time.perf_counter()
            conn.execute(f"SELECT pg_prewarm('{index_name}')")
            prewarm_time = time.perf_counter() - prewarm_start
            print(f" done! ({prewarm_time:.1f}s)")
        except psycopg.Error as e:
            print(f" failed! ({e.diag.message_primary})")
            self.debug_log(f"Prewarm failed: {e}")

        finally:
            conn.close()

    @staticmethod
    def _get_metric_operator(metric: str) -> str:
        return _metric_op(metric)

    @staticmethod
    def _get_metric_func(metric: str) -> str:
        return _metric_func(metric)

    @staticmethod
    def process_batch(args):
        """Run a worker batch. Dispatches on tuple length: HNSW uses the
        upstream 8-tuple shape unchanged; IVFFlat / BQ-rerank use a
        9-tuple that carries the SQL, bind kind, and per-benchmark GUCs."""
        if len(args) == 8:
            # HNSW path: upstream shape, byte-identical body.
            test, answer, top, metric_ops, url, table_name, ef_search, warmup_n = args

            conn = psycopg.connect(url)
            pgvector.psycopg.register_vector(conn)
            conn.execute(f"SET hnsw.ef_search={ef_search}")
            conn.execute("SET enable_seqscan = off")

            query_sql = (
                f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s "
                f"LIMIT {top}"
            )

            cursor = conn.cursor()

            if warmup_n:
                n_test = len(test)
                for j in range(warmup_n):
                    cursor.execute(query_sql, (test[j % n_test],))
                    cursor.fetchall()

            results = []
            for query, ground_truth in zip(test, answer):
                start = time.perf_counter()
                cursor.execute(query_sql, (query,))
                result = cursor.fetchall()
                end = time.perf_counter()

                result_ids = {p[0] for p in result[:top]}
                gt_ids = ground_truth[:top]
                ground_truth_ids = set(
                    gt_ids.tolist() if hasattr(gt_ids, "tolist") else gt_ids
                )
                hit = len(result_ids & ground_truth_ids)
                results.append((hit, (start, end)))

            cursor.close()
            conn.close()
            return results

        # IVFFlat / BQ-rerank dispatch tuple.
        (test, answer, top, query_sql, bind_kind, rerank_limit, gucs,
         url, warmup_n) = args

        conn = psycopg.connect(url)
        pgvector.psycopg.register_vector(conn)
        for stmt in gucs:
            conn.execute(stmt)

        cursor = conn.cursor()

        if bind_kind == "two_stage":
            bind = lambda q: (q, rerank_limit, q)
        else:
            bind = lambda q: (q,)

        if warmup_n:
            n_test = len(test)
            for j in range(warmup_n):
                cursor.execute(query_sql, bind(test[j % n_test]))
                cursor.fetchall()

        results = []
        for query, ground_truth in zip(test, answer):
            start = time.perf_counter()
            cursor.execute(query_sql, bind(query))
            result = cursor.fetchall()
            end = time.perf_counter()

            result_ids = {p[0] for p in result[:top]}
            gt_ids = ground_truth[:top]
            ground_truth_ids = set(
                gt_ids.tolist() if hasattr(gt_ids, "tolist") else gt_ids
            )
            hit = len(result_ids & ground_truth_ids)
            results.append((hit, (start, end)))

        cursor.close()
        conn.close()
        return results

    def parallel_bench(self, name, table_name, dataset, metric, top,
                        query_clients, benchmark):
        # make_batch_args (called via super().parallel_bench → make_batch_args)
        # only receives test/answer arrays, but BQ-rerank's query template
        # needs `dim` from dataset. Stash here for make_batch_args to read.
        self._batch_dataset = dataset
        try:
            return super().parallel_bench(
                name, table_name, dataset, metric, top, query_clients, benchmark,
            )
        finally:
            self._batch_dataset = None

    def make_batch_args(self, test, answer, top, metric, table_name, benchmark,
                        warmup_n=0):
        """Build the worker args tuple. HNSW path emits the upstream-shape
        8-tuple; IVFFlat variants emit a 9-tuple with SQL + GUCs baked in."""
        metric_ops = _metric_op(metric)
        if self.spec.index_type == "hnsw":
            return (
                test,
                answer,
                top,
                metric_ops,
                self.url,
                table_name,
                benchmark["efSearch"],
                warmup_n,
            )

        # warmup_query is the canonical (sql, bind_fn) source. Discard
        # bind_fn (lambda — not picklable) and rebuild it in the worker
        # from bind_kind + rerank_limit.
        dataset = self._batch_dataset
        query_sql, _bind_fn = self.warmup_query(
            table_name, dataset, metric_ops, top, benchmark,
        )
        rerank_limit = 0
        if self.spec.bind_kind == "two_stage":
            rerank_limit = top * _resolve_rerank_amp(benchmark)
        return (
            test,
            answer,
            top,
            query_sql,
            self.spec.bind_kind,
            rerank_limit,
            self.spec.session_gucs(benchmark),
            self.url,
            warmup_n,
        )

    def apply_session_guc(self, conn, benchmark):
        for stmt in self.spec.session_gucs(benchmark):
            conn.execute(stmt)

    def warmup_query(self, table_name, dataset, metric_ops, top, benchmark):
        return self.spec.query_template(
            table_name, dataset, metric_ops, top, benchmark
        )

    def create_index(self, suite_name: str, table_name: str, dataset: dict) -> None:
        """Create the pgvector index for this suite's IndexSpec
        (HNSW, IVFFlat, or IVFFlat-BQ-Rerank)."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        config = self.config[suite_name]
        pg_parallel_workers = config["pg_parallel_workers"]
        maintenance_work_mem = config.get("maintenance_work_mem")

        if self.spec.index_type == "hnsw":
            num_vectors = dataset.get("num", 0)
            dim = dataset.get("dim", 0)
            m = config["m"]
            if num_vectors and dim:
                est_bytes = _estimate_hnsw_graph_memory(num_vectors, dim, m)
                est_gb = est_bytes / (1024 ** 3)
                est_mwm = f"{int(est_gb + 1)}GB"
                est_idx_bytes = _estimate_hnsw_index_size(num_vectors, dim, m)
                est_idx_gb = est_idx_bytes / (1024 ** 3)
                print(f"Estimated HNSW graph memory: {est_gb:.1f} GB "
                      f"(recommended maintenance_work_mem >= '{est_mwm}')")
                print(f"Estimated on-disk index size: {est_idx_gb:.1f} GB "
                      f"(recommended shared_buffers >= '{int(est_idx_gb + 1)}GB' for query serving)")
        else:
            self.results[suite_name]["lists"] = _ivfflat_resolve_lists(
                config, dataset
            )

        if self.debug:
            self.spec.debug_print(config, dataset)

        conn = self.create_connection()
        start_time = time.perf_counter()

        if maintenance_work_mem:
            conn.execute(f"SET maintenance_work_mem TO '{maintenance_work_mem}'")
        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")
        conn.execute(self.spec.create_index_sql(table_name, config, dataset))

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time

        event.set()
        index_monitor_thread.join()

        print(f"Index build time: {build_time}s")

        conn.execute("CHECKPOINT")
        conn.close()
        print("Index built successfully.")

    def sequential_bench(self, name, table_name, conn, metric, top, benchmark, dataset):
        self.apply_session_guc(conn, benchmark)
        metric_ops = _metric_op(metric)

        self.debug_log(
            f"Benchmark config: {benchmark}, metric={metric}, "
            f"metric_ops={metric_ops}"
        )

        self.warmup_for_benchmark(
            conn, table_name, dataset, metric_ops, top, name,
            benchmark=benchmark,
        )

        return super().sequential_bench(
            name, table_name, conn, metric_ops, top, benchmark, dataset
        )

    def print_summary_table(self, suite_name: str):
        """HNSW renders via the upstream base method (byte-identical to
        upstream). IVFFlat / BQ-rerank need columns the base can't render
        (Probes, Rerank Amp), so they go through a spec-driven helper."""
        if self.spec.index_type == "hnsw":
            return super().print_summary_table(suite_name)

        benchmarks = self.config[suite_name].get("benchmarks", {})
        results = self.results.get(suite_name, {})
        if not benchmarks:
            return
        self._print_summary_table_for_spec(suite_name, benchmarks, results)

    def _print_summary_table_for_spec(self, suite_name, benchmarks, results):
        """Render a summary table using `self.spec.bench_param_columns`.
        Used only by the IVFFlat / BQ-rerank specs."""
        present_keys = [
            (key, header)
            for key, header in self.spec.bench_param_columns
            if any(key in b for b in benchmarks.values())
        ]
        if not present_keys:
            return

        param_headers = [h for _, h in present_keys]
        result_headers = ["Recall", "QPS", "P50 (ms)", "P99 (ms)"]
        all_headers = param_headers + result_headers

        rows = []
        for name, benchmark in benchmarks.items():
            r = results.get(name, {})
            if "recall" not in r:
                continue
            row = [str(benchmark.get(key, "N/A")) for key, _ in present_keys]
            row += [
                f"{r['recall']:.4f}",
                f"{r['qps']:.2f}",
                f"{r['p50_latency']:.2f}",
                f"{r['p99_latency']:.2f}",
            ]
            rows.append(row)

        if not rows:
            return

        widths = [len(h) for h in all_headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def fmt_row(cells):
            return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

        sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
        sb = results.get("shared_buffers", "N/A")
        idx_size = results.get("index_size", "N/A")
        qc = results.get("query_clients", 1)

        print(f"\n{'=' * len(sep)}")
        print(f"  Results Summary: {suite_name}")
        print(f"  shared_buffers: {sb} | clients: {qc} | index_size: {idx_size}")
        print(f"{'=' * len(sep)}")
        print(fmt_row(all_headers))
        print(sep)
        for row in rows:
            print(fmt_row(row))
        print()

    def generate_markdown_result(self):
        """Generate benchmark results with charts and consolidated CSV."""
        self.debug_log(f"Results: {self.results}")

        results_manager = ResultsManager()

        for suite_name in self.config:
            system_metrics, pg_stats, dashboard_path = self.get_monitoring_data(suite_name)
            kwargs = dict(
                suite_type=self.spec.suite_type,
                config={suite_name: self.config[suite_name]},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
            )
            # HNSW renders via the legacy "pgvector" branch in results.py
            # and ignores the new column kwargs. IVFFlat variants take the
            # new branch and consume them.
            if self.spec.index_type != "hnsw":
                kwargs["config_columns"] = list(self.spec.config_columns)
                kwargs["bench_columns"] = list(self.spec.bench_param_columns)
            results_manager.process_suite_results(**kwargs)


def main():
    """Main entry point for pgvector benchmark suite."""
    parser = build_arg_parse()
    args = parser.parse_args()

    test_suite = TestSuite(
        suite_file=args.suite,
        url=args.url,
        devices=args.devices,
        chunk_size=args.chunk_size,
        skip_add_embeddings=args.skip_add_embeddings,
        centroids=args.centroids_file,
        centroids_table=args.centroids_table,
        skip_index_creation=args.skip_index_creation,
        query_clients=args.query_clients,
        max_load_threads=args.max_load_threads,
        debug=args.debug,
        overwrite_table=args.overwrite_table,
        debug_single_query=args.debug_single_query,
        build_only=args.build_only,
        max_queries=args.max_queries,
        warmup=args.warmup,
    )

    test_suite.run()
    print("Test suite completed.")


if __name__ == "__main__":
    main()
