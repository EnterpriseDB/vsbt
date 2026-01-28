"""
PGPU Benchmark Suite

Benchmarks GPU-accelerated vector index building using the PGPU extension
with VectorChord for PostgreSQL.
"""

import argparse
import re
import time

import psycopg
import pgvector.psycopg

import common
from results import ResultsManager


def build_arg_parse():
    """Build argument parser for PGPU benchmark suite."""
    parser = argparse.ArgumentParser(description="PGPU Benchmark Suite")
    common.build_arg_parse(parser)
    return parser


class TestSuite(common.TestSuite):
    """
    Test suite for PGPU GPU-accelerated vector indexing.

    Uses the pgpu extension to build IVF indexes on GPU,
    then queries using VectorChord's vchordrq index type.
    """

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
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgpu")
        conn.close()
        self.debug_log("Extensions initialized successfully.")

    def prewarm_index(self, table_name: str):
        """Prewarm the index into memory for consistent benchmarking."""
        conn = self.create_connection()
        print("Prewarming the index...", end="", flush=True)
        try:
            conn.execute(
                f"SELECT vchordrq_prewarm('{table_name}_pgpu_ext'::regclass)"
            )
            print(" done!")
        except psycopg.Error as e:
            print(f" failed! ({e.diag.message_primary})")
            self.debug_log(f"Prewarm failed: {e}")
        finally:
            conn.close()

    def make_handler(self, suite_name: str):
        """
        Create a notice handler to capture clustering time from PGPU output.

        Matches messages like: "Training complete (123.45s). Building VectorChord Index..."
        """
        pattern = re.compile(
            r"Training complete\s*\((.*?)\)\.\s*Building VectorChord Index",
            re.IGNORECASE,
        )

        def handler(notice):
            print(notice.message_primary)
            match = pattern.search(notice.message_primary)
            if match:
                self.results[suite_name]["clustering_time"] = match.group(1)

        return handler

    def create_index(self, suite_name: str, table_name: str, dataset: dict = None):
        """Create a GPU-accelerated IVF index using PGPU."""
        event, index_monitor_thread = super().create_index(
            suite_name, table_name, dataset
        )

        # Load configuration
        config = self.config[suite_name]
        pg_parallel_workers = config["pg_parallel_workers"]
        lists = config["lists"]
        sampling_factor = config["samplingFactor"]
        residual_quantization = config["residual_quantization"]
        batch_size = config["batchSize"]
        distance = config["metric"]
        kmeans_nredo = config.get("kmeans_n_redo", 3)
        kmeans_n_iter = config.get("kmeans_n_iter", 30)
        random_sampling = config.get("random_sampling", "true")

        # Normalize metric name (pgpu uses "ip" for inner product)
        distance = "ip" if distance == "dot" else distance
        spherical_bool = "true" if distance in ("ip", "cos") else "false"

        if self.debug:
            print(f"\n🔧 Index Configuration (pgpu):")
            print(f"    • Lists:           {lists}")
            print(f"    • Sampling Factor: {sampling_factor}")
            print(f"    • Batch Size:      {batch_size:,}")
            print(f"    • Distance:        {distance}")
            print(f"    • Spherical:       {spherical_bool}")
            print(f"    • Residual Quant:  {residual_quantization}")
            print()

        self.results[suite_name]["lists"] = lists

        conn = self.create_connection()
        conn.add_notice_handler(self.make_handler(suite_name))
        start_time = time.perf_counter()

        conn.execute(f"SET max_parallel_maintenance_workers TO {pg_parallel_workers}")
        conn.execute(f"SET max_parallel_workers TO {pg_parallel_workers}")

        conn.execute(
            f"""
            SELECT pgpu.create_vector_index_on_gpu(
                table_name => 'public.{table_name}',
                column_name => 'embedding',
                lists => ARRAY{lists},
                sampling_factor => {sampling_factor},
                batch_size => {batch_size},
                kmeans_nredo => {kmeans_nredo},
                kmeans_iterations => {kmeans_n_iter},
                distance_operator => '{distance}',
                spherical_centroids => '{spherical_bool}',
                residual_quantization => {residual_quantization},
                random_sampling => {random_sampling}
            )
            """
        )

        build_time = int(round(time.perf_counter() - start_time))
        self.results[suite_name]["index_build_time"] = build_time
        print(f"Index build time: {build_time}s")

        conn.execute("CHECKPOINT")
        conn.close()
        print("Index built successfully.")

        event.set()
        index_monitor_thread.join()

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
            f"nprob: {benchmark['nprob']}, epsilon: {benchmark['epsilon']}, "
            f"metric: {metric}, metric_ops: {metric_ops}"
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
                suite_type="pgpu",
                config={suite_name: self.config[suite_name]},
                results={suite_name: self.results.get(suite_name, {})},
                query_clients=self.query_clients,
                system_metrics=system_metrics,
                pg_stats=pg_stats,
                system_dashboard_path=dashboard_path,
            )


def main():
    """Main entry point for PGPU benchmark suite."""
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
