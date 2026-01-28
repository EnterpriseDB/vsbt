"""
VectorChord Benchmark Suite

Benchmarks vector search using the VectorChord extension with IVF indexes
(vchordrq) for PostgreSQL.
"""

import argparse
import time

import psycopg
import pgvector.psycopg

import common
from results import ResultsManager


def build_arg_parse():
    """Build argument parser for VectorChord benchmark suite."""
    parser = argparse.ArgumentParser(description="VectorChord Benchmark Suite")
    common.build_arg_parse(parser)
    return parser


class TestSuite(common.TestSuite):
    """
    Test suite for VectorChord IVF indexing.

    Uses the VectorChord extension to build IVF indexes with optional
    residual quantization for approximate nearest neighbor searches.
    """

    METRIC_OPS = {
        "l2": "vector_l2_ops",
        "euclidean": "vector_l2_ops",
        "cos": "vector_cosine_ops",
        "ip": "vector_ip_ops",
        "dot": "vector_ip_ops",
    }

    @staticmethod
    def process_batch(args):
        """Process a batch of queries in parallel."""
        test, answer, top, metric_ops, url, table_name, nprob, epsilon = args

        conn = psycopg.connect(url)
        pgvector.psycopg.register_vector(conn)
        conn.execute("SET jit=false")
        conn.execute(f'SET vchordrq.probes="{nprob}"')
        conn.execute(f"SET vchordrq.epsilon={epsilon}")

        query_sql = f"SELECT id FROM {table_name} ORDER BY embedding {metric_ops} %s LIMIT {top}"

        results = []
        cursor = conn.cursor()
        for query, ground_truth in zip(test, answer):
            start = time.perf_counter()
            cursor.execute(query_sql, (query,))
            result = cursor.fetchall()
            end = time.perf_counter()

            result_ids = {p[0] for p in result[:top]}
            gt_ids = ground_truth[:top]
            ground_truth_ids = set(gt_ids.tolist() if hasattr(gt_ids, "tolist") else gt_ids)
            hit = len(result_ids & ground_truth_ids)
            results.append((hit, (start, end)))

        cursor.close()
        conn.close()
        return results

    def make_batch_args(self, test, answer, top, metric, table_name, benchmark):
        """Prepare arguments for parallel batch processing."""
        metric_ops = self._get_metric_operator(metric)
        return (
            test,
            answer,
            top,
            metric_ops,
            self.url,
            table_name,
            benchmark["nprob"],
            benchmark["epsilon"],
        )

    @staticmethod
    def _get_metric_operator(metric: str) -> str:
        """Convert metric name to PostgreSQL operator."""
        operators = {
            "l2": "<->",
            "euclidean": "<->",
            "cos": "<=>",
            "angular": "<=>",
            "dot": "<#>",
            "ip": "<#>",
        }
        if metric not in operators:
            raise ValueError(f"Unsupported metric type: {metric}")
        return operators[metric]

    def create_connection(self):
        """Create a database connection with pgvector support."""
        conn = super().create_connection()
        pgvector.psycopg.register_vector(conn)
        return conn

    def init_ext(self, suite_name: str = None):
        """Initialize required PostgreSQL extensions."""
        conn = super().create_connection()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        """Prewarm the index into memory for consistent benchmarking."""
        conn = self.create_connection()
        print("Prewarming the index...", end="", flush=True)
        try:
            conn.execute(
                f"SELECT vchordrq_prewarm('{table_name}_embedding_idx'::regclass)"
            )
            print(" done!")
        except psycopg.Error as e:
            print(f" failed! ({e.diag.message_primary})")
            self.debug_log(f"Prewarm failed: {e}")
        finally:
            conn.close()

    def create_index(self, suite_name: str, table_name: str, dataset: dict):
        """Create an IVF index using VectorChord."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        # Load configuration
        config = self.config[suite_name]
        pg_parallel_workers = config["pg_parallel_workers"]
        lists = config["lists"]
        build_threads = config.get("build_threads", 1)
        kmeans_hierarchical = config["kmeans_hierarchical"]
        kmeans_dimension = config.get("kmeans_dimension", dataset["dim"])
        sampling_factor = config["samplingFactor"]
        residual_quantization = config["residual_quantization"]
        metric = dataset["metric"]

        if self.debug:
            print(f"\n🔧 Index Configuration (VectorChord):")
            print(f"    • Lists:           {lists}")
            print(f"    • Build Threads:   {build_threads}")
            print(f"    • K-means Hier.:   {kmeans_hierarchical}")
            print(f"    • Sampling Factor: {sampling_factor}")
            print(f"    • Residual Quant:  {residual_quantization}")
            print()

        self.results[suite_name]["lists"] = lists
        self.results[suite_name]["build_threads"] = build_threads

        metric_ops = self.METRIC_OPS[metric]
        kmeans_algo = "kmeans_algorithm.hierarchical = {}" if kmeans_hierarchical else "kmeans_algorithm.lloyd = {}"
        rq_string = "true" if residual_quantization else "false"

        # Build IVF configuration
        ivf_config = self._build_ivf_config(
            metric, lists, build_threads, kmeans_dimension,
            sampling_factor, kmeans_algo, rq_string
        )

        self.debug_log(f"Centroids source: {'external file' if self.centroids else 'external table' if self.centroids_table else 'internal'}")

        conn = self.create_connection()
        start_time = time.perf_counter()

        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")
        conn.execute(
            f"CREATE INDEX {table_name}_embedding_idx ON {table_name} "
            f"USING vchordrq (embedding {metric_ops}) WITH (options = $${ivf_config}$$)"
        )

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time
        print(f"Index build time: {build_time}s")

        conn.execute("CHECKPOINT")
        conn.close()
        print("Index built successfully.")

        event.set()
        index_monitor_thread.join()

    def _build_ivf_config(
        self,
        metric: str,
        lists: list,
        build_threads: int,
        kmeans_dimension: int,
        sampling_factor: int,
        kmeans_algo: str,
        rq_string: str,
    ) -> str:
        """Build the IVF configuration string for index creation."""
        base_config = f"""
        residual_quantization = {rq_string}
        build.pin = 2
        """

        if self.centroids:
            external_cfg = """
            [build.external]
            table = 'public.centroids'
            """
            return "\n".join([base_config, external_cfg])

        if self.centroids_table:
            external_cfg = f"""
            [build.external]
            table = '{self.centroids_table}'
            """
            return "\n".join([base_config, external_cfg])

        # Internal centroids
        spherical = "true" if metric in ("cos", "dot", "ip") else "false"
        internal_cfg = f"""
        [build.internal]
        lists = {lists}
        build_threads = {build_threads}
        spherical_centroids = {spherical}
        kmeans_dimension = {kmeans_dimension}
        sampling_factor = {sampling_factor}
        {kmeans_algo}
        """
        return "\n".join([base_config, internal_cfg])

    def sequential_bench(
        self,
        name: str,
        table_name: str,
        conn: psycopg.Connection,
        metric: str,
        top: int,
        benchmark: dict,
        dataset: dict,
    ) -> tuple[list[tuple[int, float]], str]:
        """Run sequential benchmark queries."""
        conn.execute("SET jit=false")
        conn.execute(f'SET vchordrq.probes="{benchmark["nprob"]}"')
        conn.execute(f"SET vchordrq.epsilon={benchmark['epsilon']}")

        metric_ops = self._get_metric_operator(metric)

        self.debug_log(
            f"Benchmark config: nprob={benchmark['nprob']}, epsilon={benchmark['epsilon']}, "
            f"metric={metric}, metric_ops={metric_ops}"
        )

        return super().sequential_bench(
            name, table_name, conn, metric_ops, top, benchmark, dataset
        )

    def generate_markdown_result(self):
        """Generate benchmark results with charts and consolidated CSV."""
        self.debug_log(f"Results: {self.results}")

        results_manager = ResultsManager()

        # Get monitoring data for each suite
        for suite_name in self.config:
            system_metrics, pg_stats, dashboard_path = self.get_monitoring_data(suite_name)

            results_manager.process_suite_results(
                suite_type="vectorchord",
                config={suite_name: self.config[suite_name]},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
            )


def main():
    """Main entry point for VectorChord benchmark suite."""
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
    )

    test_suite.run()
    print("Test suite completed.")


if __name__ == "__main__":
    main()
