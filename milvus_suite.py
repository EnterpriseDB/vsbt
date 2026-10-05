"""Milvus server benchmarks for IVF_FLAT and HNSW.

Uses VSBT datasets and reports, with explicit INT64 IDs so recall is scored
against the same ground truth as the PostgreSQL suites.
"""

import argparse
import gc
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

import common
import datasets
from monitor.system_monitor import SystemMonitor, generate_system_report, is_local_database
from results import ResultsManager


INDEX_NAME = "embedding_idx"
CONFIG_COLUMNS = (
    ("Index Type", lambda c, r: index_type(c)),
    ("Collection", lambda c, r: r.get("collection", "N/A")),
    ("Lists (nlist)", lambda c, r: str(r.get("lists", "N/A"))),
    ("M", lambda c, r: str(c.get("m", "N/A"))),
    ("EF Construction", lambda c, r: str(c.get("efConstruction", "N/A"))),
)


def positive_int(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}; got {value}")
    return value


def index_type(config):
    value = str(config.get("indexType", "")).lower().replace("_", "")
    if value not in ("ivfflat", "hnsw"):
        raise ValueError("Milvus indexType must be 'ivfflat' or 'hnsw'")
    return {"ivfflat": "IVF_FLAT", "hnsw": "HNSW"}[value]


def metric_type(metric):
    aliases = {"l2": "L2", "euclidean": "L2", "cos": "COSINE",
               "cosine": "COSINE", "angular": "COSINE", "ip": "IP", "dot": "IP"}
    try:
        return aliases[str(metric).lower()]
    except KeyError:
        raise ValueError(f"Unsupported Milvus metric: {metric!r}") from None


def build_params(config, dataset):
    if index_type(config) == "IVF_FLAT":
        lists = config.get("lists", "auto")
        if isinstance(lists, str) and lists.lower() == "auto":
            lists = min(65536, max(1, int(math.sqrt(dataset["num"]))))
        return {"nlist": positive_int(lists, "lists", 65536)}
    m = positive_int(config.get("m", 16), "m", 2048)
    if m < 2:
        raise ValueError("HNSW m must be >= 2")
    return {"M": m, "efConstruction": positive_int(
        config.get("efConstruction", 128), "efConstruction")}


def search_params(config, benchmark, dataset):
    if index_type(config) == "IVF_FLAT":
        value = positive_int(benchmark.get("probes"), "probes")
        if value > build_params(config, dataset)["nlist"]:
            raise ValueError("probes must be <= lists")
        params = {"nprobe": value}
    else:
        value = positive_int(benchmark.get("efSearch"), "efSearch")
        if value < config["top"]:
            raise ValueError("efSearch must be >= top")
        params = {"ef": value}
    return {"metric_type": metric_type(config["metric"]), "params": params}


def collection_name(config, dataset):
    name = config.get("collection")
    if name is None:
        suffix = "_".join(f"{k}_{v}" for k, v in build_params(config, dataset).items())
        name = f"vsbt_{config['dataset']}_{index_type(config).lower()}_{suffix}"
        name = name.replace("-", "_")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", name):
        raise ValueError(f"Invalid Milvus collection name: {name!r}")
    return name


def embedding_batches(dataset, batch_size):
    """Bound memory/RPC size and preserve both explicit and positional IDs."""
    train = dataset["train"]
    if hasattr(train, "shape"):
        # Slice HDF5/mmap arrays in bulk instead of doing one disk read per row.
        for offset in range(0, len(train), batch_size):
            vectors = np.asarray(train[offset:offset + batch_size], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != dataset["dim"] or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid embeddings near ID {offset}")
            yield [{"id": offset + i, "embedding": vector}
                   for i, vector in enumerate(vectors.tolist())]
        return
    rows = iter(train)  # parquet and multipart loaders yield (id, vector)
    batch = []
    for row_id, embedding in rows:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (dataset["dim"],) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid embedding at ID {row_id}: expected {dataset['dim']} finite values")
        batch.append({"id": int(row_id), "embedding": vector.tolist()})
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def close_training_data(train):
    if hasattr(train, "close"):
        train.close()
    elif hasattr(train, "file"):
        train.file.close()


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Milvus IVF_FLAT / HNSW Benchmark Suite")
    common.build_arg_parse(parser)
    parser.set_defaults(url="http://localhost:19530", chunk_size=1000)
    # Common options without Milvus equivalents should fail instead of being ignored.
    for action in list(parser._actions):
        if action.dest in ("centroids_file", "centroids_table"):
            parser._remove_action(action)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
            for option in action.option_strings:
                del parser._option_string_actions[option]
    for action in parser._actions:
        if action.dest == "url":
            action.help = "Milvus server URI (default: http://localhost:19530)"
        elif action.dest == "overwrite_table":
            action.help = "Drop and recreate this benchmark's Milvus collection"
        elif action.dest == "chunk_size":
            action.help = "Maximum rows per insert RPC (also capped by vector dimension)"
        elif action.dest == "warmup":
            action.help = "Query warmup: auto, off, or an integer N (excluded from metrics)"
        elif action.dest == "devices":
            action.help = "Block devices to monitor on a local Milvus host"
    parser.add_argument("--db-name", default=os.getenv("MILVUS_DB_NAME", "default"))
    parser.add_argument("--timeout", type=float, default=86400,
                        help="RPC/build timeout in seconds (default: 86400)")
    return parser


class TestSuite:
    def __init__(self, suite_file, url="http://localhost:19530", devices=None,
                 chunk_size=1000, skip_add_embeddings=False, skip_index_creation=False,
                 query_clients=1, max_queries=None, max_load_threads=4, debug=False,
                 overwrite_table=False, debug_single_query=False, build_only=False,
                 warmup="auto", db_name="default", timeout=86400):
        if not url.startswith(("http://", "https://", "tcp://")):
            raise ValueError("Use a Milvus server URI; Milvus Lite is not supported for these benchmarks")
        self.config = common.load_suite_config(suite_file)
        self.url, self.db_name, self.timeout = url, db_name, timeout
        self.token = os.getenv("MILVUS_TOKEN", "")
        self.devices = devices
        self.chunk_size = positive_int(chunk_size, "chunk-size")
        self.query_clients = positive_int(query_clients, "query-clients")
        self.max_load_threads = positive_int(max_load_threads, "max-load-threads")
        self.max_queries = None if max_queries is None else positive_int(max_queries, "max-queries")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.skip_add_embeddings, self.skip_index_creation = skip_add_embeddings, skip_index_creation
        self.overwrite_table, self.build_only = overwrite_table, build_only
        self.debug, self.debug_single_query = debug, debug_single_query
        self.warmup = common.TestSuite._parse_warmup(warmup)
        self.results = {}
        if overwrite_table and skip_add_embeddings:
            raise ValueError("--overwrite-table requires data loading")
        if overwrite_table and skip_index_creation:
            raise ValueError("--overwrite-table requires index creation")
        for name, config in self.config.items():
            if config.get("vectorType", "vector") != "vector":
                raise ValueError(f"{name}: Milvus supports float32 vector benchmarks only")
            dataset_name = config.get("dataset")
            if dataset_name not in datasets.DATASETS:
                raise ValueError(f"{name}: unknown dataset {dataset_name!r}")
            ds = datasets.DATASETS[dataset_name]
            config.setdefault("metric", ds["metric"])
            config.setdefault("top", 10)
            positive_int(config["top"], "top", 16384)
            if metric_type(config["metric"]) != metric_type(ds["metric"]):
                raise ValueError(f"{name}: metric must match the dataset's ground truth")
            resolved = build_params(config, ds)
            if index_type(config) == "HNSW":
                config.setdefault("m", resolved["M"])
                config.setdefault("efConstruction", resolved["efConstruction"])
            collection_name(config, ds)
            for benchmark in config.get("benchmarks", {}).values():
                search_params(config, benchmark, ds)

    def create_connection(self):
        # Lazy import keeps offline checks and PostgreSQL suites independent.
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise RuntimeError("Install Milvus support with: pip install -r requirements-milvus.txt") from None
        return MilvusClient(uri=self.url, token=self.token, db_name=self.db_name, timeout=self.timeout)

    def add_embeddings(self, client, collection, ds):
        from pymilvus import DataType

        if client.has_collection(collection_name=collection, timeout=self.timeout):
            if not self.overwrite_table:
                raise ValueError(f"Collection {collection} already exists; use --skip-add-embeddings "
                                 "to reuse it or --overwrite-table to reload it")
            client.drop_collection(collection_name=collection, timeout=self.timeout)
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=ds["dim"])
        client.create_collection(collection_name=collection, schema=schema,
                                 consistency_level="Strong", timeout=self.timeout)
        # Limit each request to ~8 MiB of vector payload, safely below the RPC limit.
        batch_size = min(self.chunk_size, max(1, (8 * 1024 * 1024) // (4 * ds["dim"])))
        start = time.perf_counter()
        count = 0
        def insert(batch):
            response = client.insert(collection_name=collection, data=batch, timeout=self.timeout)
            if response["insert_count"] != len(batch):
                raise RuntimeError("Milvus did not insert the entire batch")
            return len(batch)

        # Bound pending work; executor.map would eagerly consume a billion-row generator.
        with ThreadPoolExecutor(max_workers=self.max_load_threads) as pool:
            pending = []
            with tqdm(total=ds["num"], desc="Loading Milvus embeddings") as progress:
                for batch in embedding_batches(ds, batch_size):
                    pending.append(pool.submit(insert, batch))
                    if len(pending) >= self.max_load_threads:
                        loaded = pending.pop(0).result()
                        count += loaded
                        progress.update(loaded)
                for future in pending:
                    loaded = future.result()
                    count += loaded
                    progress.update(loaded)
        client.flush(collection_name=collection, timeout=self.timeout)
        if count != ds["num"]:
            raise ValueError(f"Dataset yielded {count} rows, expected {ds['num']}")
        return time.perf_counter() - start

    def verify_collection(self, client, collection, ds):
        description = client.describe_collection(collection_name=collection, timeout=self.timeout)
        fields = {field["name"]: field for field in description["fields"]}
        vector = fields.get("embedding", {})
        primary = fields.get("id", {})
        # DataType.INT64 = 5, FLOAT_VECTOR = 101 (wire enum values).
        if (int(vector.get("type", 0)) != 101 or
                int(vector.get("params", {}).get("dim", 0)) != ds["dim"] or
                int(primary.get("type", 0)) != 5 or not primary.get("is_primary") or
                description.get("auto_id", False)):
            raise ValueError(f"Collection {collection} must have explicit INT64 id and FLOAT_VECTOR embedding({ds['dim']})")
        count = int(client.get_collection_stats(collection_name=collection, timeout=self.timeout)["row_count"])
        if count != ds["num"]:
            raise ValueError(f"Collection {collection} has {count} rows; expected {ds['num']}")

    def create_index(self, client, collection, config, ds):
        # Loading existing data may leave this collection loaded. Release before replacing indexes.
        client.release_collection(collection_name=collection, timeout=self.timeout)
        for name in client.list_indexes(collection_name=collection, timeout=self.timeout):
            description = client.describe_index(collection_name=collection, index_name=name, timeout=self.timeout)
            if description.get("field_name") == "embedding":
                client.drop_index(collection_name=collection, index_name=name, timeout=self.timeout)
        params = client.prepare_index_params()
        params.add_index(field_name="embedding", index_name=INDEX_NAME,
                         index_type=index_type(config), metric_type=metric_type(config["metric"]),
                         params=build_params(config, ds))
        start = time.perf_counter()
        client.create_index(collection_name=collection, index_params=params, sync=True, timeout=self.timeout)
        return time.perf_counter() - start

    def verify_index(self, client, collection, config, ds):
        description = client.describe_index(collection_name=collection, index_name=INDEX_NAME, timeout=self.timeout)
        if not description:
            raise ValueError(f"Missing index {collection}.{INDEX_NAME}; run without --skip-index-creation")
        actual = {**description.get("params", {}), **description}
        expected = {"index_type": index_type(config), "metric_type": metric_type(config["metric"]),
                    **build_params(config, ds)}
        for key, value in expected.items():
            if str(actual.get(key)) != str(value):
                raise ValueError(f"Existing index {collection}.{INDEX_NAME}: expected {key}={value}, "
                                 f"got {actual.get(key)!r}")
        if description.get("state") != "Finished" or int(description.get("pending_index_rows", 0)):
            raise ValueError(f"Index {collection}.{INDEX_NAME} is not fully built: {description}")
        return description

    def search(self, client, collection, query, top, params):
        return client.search(collection_name=collection, anns_field="embedding",
                             data=[query], limit=top, search_params=params, timeout=self.timeout,
                             consistency_level="Strong")[0]

    def warmup_queries(self, client, collection, queries, top, params):
        if self.warmup == "off" or self.debug_single_query:
            return 0
        target = self.warmup if isinstance(self.warmup, int) else common.WARMUP_MAX
        history = []
        count, start = 0, time.perf_counter()
        while count < target:
            self.search(client, collection, queries[count % len(queries)], top, params)
            count += 1
            if count % common.WARMUP_CHUNK:
                continue
            now = time.perf_counter()
            history.append(common.WARMUP_CHUNK / max(now - start, 1e-12))
            start = now
            window = common.WARMUP_CHUNKS_PER_WINDOW
            if self.warmup == "auto" and count >= common.WARMUP_MIN and len(history) >= 2 * window:
                prior = sum(history[-2 * window:-window])
                recent = sum(history[-window:])
                if abs(recent - prior) / prior < common.WARMUP_QPS_TOLERANCE:
                    break
        return count

    def query_batch(self, collection, queries, answers, top, params, warmup_n):
        client = self.create_connection()
        try:
            for i in range(warmup_n):
                self.search(client, collection, queries[i % len(queries)], top, params)
            results = []
            for query, ground_truth in zip(queries, answers):
                start = time.perf_counter()
                hits = self.search(client, collection, query, top, params)
                end = time.perf_counter()
                ids = {int(hit["id"]) for hit in hits[:top]}
                results.append((len(ids & set(ground_truth[:top])), (start, end)))
            return results
        finally:
            client.close()

    def run_benchmarks(self, client, collection, name, config, ds):
        queries = ds["test"][:self.max_queries]
        answers = ds["neighbors"][:self.max_queries]
        if len(queries) == 0 or len(answers) != len(queries) or answers.shape[1] < config["top"]:
            raise ValueError("Dataset must have queries and at least top ground-truth IDs per query")
        if self.debug_single_query:
            queries = np.repeat(queries[:1], len(queries), axis=0)
            answers = np.repeat(answers[:1], len(answers), axis=0)
        # Convert once outside timed loops; one RPC per query, like the SQL suites.
        queries = np.asarray(queries, dtype=np.float32).tolist()
        for bench_name, benchmark in config.get("benchmarks", {}).items():
            params = search_params(config, benchmark, ds)
            warmup_n = self.warmup_queries(client, collection, queries, config["top"], params)
            print(f"[warmup] {bench_name}: n={warmup_n}")
            with ThreadPoolExecutor(max_workers=self.query_clients) as pool:
                futures = [pool.submit(self.query_batch, collection, queries, answers,
                                       config["top"], params, warmup_n if self.query_clients > 1 else 0)
                           for _ in range(self.query_clients)]
                measured = [item for future in futures for item in future.result()]
            recall, qps, p50, p99 = common.calculate_metrics(
                measured, config["top"], len(queries), self.query_clients)
            self.results[name][bench_name] = {
                "recall": recall, "qps": qps, "p50_latency": p50, "p99_latency": p99}
            print(f"{bench_name}: Recall={recall:.4f} QPS={qps:.2f} P50={p50:.2f}ms P99={p99:.2f}ms")

    def run_suite(self, name):
        config = self.config[name]
        ds = datasets.get_dataset(config["dataset"])
        collection = collection_name(config, ds)
        result = self.results[name] = {"query_clients": self.query_clients, "collection": collection,
                                      "index_type": index_type(config), "index_size": "N/A"}
        if index_type(config) == "IVF_FLAT":
            result["lists"] = build_params(config, ds)["nlist"]
        monitor = None
        client = None
        try:
            if is_local_database(self.url):
                result["system_report"] = generate_system_report()
                monitor = SystemMonitor(results_dir=f"./results/{name}", devices=self.devices or [])
                monitor.start()
            client = self.create_connection()
            if not self.skip_add_embeddings:
                if monitor:
                    monitor.mark_phase("load_start")
                result["load_time"] = self.add_embeddings(client, collection, ds)
                if monitor:
                    monitor.mark_phase("load_end")
            self.verify_collection(client, collection, ds)
            train = ds.pop("train")
            close_training_data(train)
            del train
            gc.collect()
            if not self.skip_index_creation:
                if self.debug:
                    print(f"Building {index_type(config)} on {collection}: {build_params(config, ds)}")
                if monitor:
                    monitor.mark_phase("index_start")
                result["index_build_time"] = self.create_index(client, collection, config, ds)
                if monitor:
                    monitor.mark_phase("index_end")
            result["milvus_index"] = self.verify_index(client, collection, config, ds)
            if not self.build_only:
                start = time.perf_counter()
                client.load_collection(collection_name=collection, timeout=self.timeout)
                result["collection_load_time"] = time.perf_counter() - start
                if monitor:
                    monitor.mark_phase("benchmark_start")
                self.run_benchmarks(client, collection, name, config, ds)
                if monitor:
                    monitor.mark_phase("benchmark_end")
        finally:
            if client:
                client.close()
            if monitor:
                monitor.stop()
            train = ds.get("train")
            close_training_data(train)
        manager = ResultsManager()
        columns = (("probes", "Probes (nprobe)"),) if index_type(config) == "IVF_FLAT" else (("efSearch", "EF Search"),)
        manager.process_suite_results(
            suite_type="milvus", config={name: config}, results={name: result},
            query_clients=self.query_clients, config_columns=CONFIG_COLUMNS, bench_columns=columns,
            system_metrics=monitor.format_for_report() if monitor else None,
            system_dashboard_path=monitor.generate_dashboard(name) if monitor else None)
        if monitor:
            monitor.save_csv(f"{name}_system_metrics.csv")

    def run(self):
        for name in self.config:
            print(f"Running Milvus suite: {name}")
            self.run_suite(name)


def main():
    args = vars(build_arg_parse().parse_args())
    args["suite_file"] = args.pop("suite")
    TestSuite(**args).run()


if __name__ == "__main__":
    main()
