"""Offline lifecycle/recall/report checks using a Milvus client double.

Run after installing requirements-milvus.txt:
    python -m unittest test_milvus_suite.py
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import yaml
from pymilvus import MilvusClient

import milvus_suite as mv
from results import ResultsManager


class FakeClient:
    """Use real SDK schema/index objects and exact search over inserted rows."""

    create_schema = staticmethod(MilvusClient.create_schema)
    prepare_index_params = staticmethod(MilvusClient.prepare_index_params)

    def __init__(self):
        self.collections = {}
        self.events = []

    def has_collection(self, collection_name, **kwargs):
        return collection_name in self.collections

    def create_collection(self, collection_name, schema, **kwargs):
        self.collections[collection_name] = {"schema": schema.to_dict(), "rows": [], "indexes": {}}
        self.events.append("create_collection")

    def drop_collection(self, collection_name, **kwargs):
        del self.collections[collection_name]
        self.events.append("drop_collection")

    def insert(self, collection_name, data, **kwargs):
        self.collections[collection_name]["rows"].extend(data)
        self.events.append("insert")
        return {"insert_count": len(data)}

    def flush(self, **kwargs):
        self.events.append("flush")

    def describe_collection(self, collection_name, **kwargs):
        return self.collections[collection_name]["schema"]

    def get_collection_stats(self, collection_name, **kwargs):
        return {"row_count": len(self.collections[collection_name]["rows"])}

    def release_collection(self, **kwargs):
        self.events.append("release")

    def list_indexes(self, collection_name, **kwargs):
        return list(self.collections[collection_name]["indexes"])

    def create_index(self, collection_name, index_params, sync, **kwargs):
        assert sync is True
        for param in index_params:
            self.collections[collection_name]["indexes"][param.index_name] = {
                **param.get_index_configs(), "field_name": param.field_name, "state": "Finished",
                "pending_index_rows": 0}
        self.events.append("create_index")

    def describe_index(self, collection_name, index_name, **kwargs):
        return self.collections[collection_name]["indexes"][index_name]

    def drop_index(self, collection_name, index_name, **kwargs):
        del self.collections[collection_name]["indexes"][index_name]
        self.events.append("drop_index")

    def load_collection(self, **kwargs):
        self.events.append("load")

    def search(self, collection_name, data, limit, search_params, **kwargs):
        self.events.append("search")
        rows = self.collections[collection_name]["rows"]
        vectors = np.array([row["embedding"] for row in rows])
        query = np.array(data[0])
        if search_params["metric_type"] == "L2":
            scores = np.sum((vectors - query) ** 2, axis=1)
        elif search_params["metric_type"] == "COSINE":
            scores = -(vectors @ query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
        else:
            scores = -(vectors @ query)
        return [[{"id": rows[i]["id"], "distance": float(scores[i])}
                 for i in np.argsort(scores)[:limit]]]

    def close(self):
        self.events.append("close")


class MilvusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.metadata = {"metric": "cos", "dim": 2, "num": 4}
        self.train = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=np.float32)
        self.ids = [17, 3, 99, 8]
        self.client = FakeClient()
        self.addCleanup(patch.stopall)
        patch.dict(mv.datasets.DATASETS, {"fixture": self.metadata}).start()

    def config(self, kind="ivfflat"):
        return {"indexType": kind, "dataset": "fixture", "metric": "cos",
                "lists": 2, "m": 16, "efConstruction": 128, "top": 1,
                "benchmarks": {"point": {"probes": 2} if kind == "ivfflat" else {"efSearch": 10}}}

    def suite(self, config=None, **kwargs):
        path = self.path / "suite.yaml"
        path.write_text(yaml.safe_dump({"milvus-fixture": config or self.config()}))
        return mv.TestSuite(str(path), url="http://remote.example:19530", warmup="off", **kwargs)

    def dataset(self):
        return {**self.metadata, "train": iter(zip(self.ids, self.train)),
                "test": self.train, "neighbors": np.array(self.ids).reshape(-1, 1)}

    def test_lifecycle_recall_parallel_and_report(self):
        for kind in ("ivfflat", "hnsw"):
            for clients in (1, 2):
                with self.subTest(kind=kind, clients=clients):
                    config = self.config(kind)
                    if kind == "hnsw":
                        config.pop("lists")
                    else:
                        config.pop("m")
                        config.pop("efConstruction")
                    suite = self.suite(config, query_clients=clients, max_queries=2)
                    output = self.path / f"{kind}-{clients}"
                    with patch.object(suite, "create_connection", return_value=self.client), \
                         patch.object(mv.datasets, "get_dataset", side_effect=lambda _: self.dataset()), \
                         patch.object(mv, "ResultsManager", side_effect=lambda: ResultsManager(str(output))), \
                         contextlib.redirect_stdout(io.StringIO()):
                        suite.run()
                    result = suite.results["milvus-fixture"]
                    self.assertEqual(result["point"]["recall"], 1)
                    self.assertGreater(result["point"]["qps"], 0)
                    self.assertEqual([r["id"] for r in self.client.collections[result["collection"]]["rows"]], self.ids)
                    events = self.client.events
                    self.assertLess(events.index("flush"), events.index("create_index"))
                    self.assertLess(events.index("create_index"), events.index("load"))
                    run = json.loads(next(output.rglob("run.json")).read_text())
                    self.assertEqual(run["metadata"]["suite_type"], "milvus")
                    with (output / "all_results.csv").open() as f:
                        row = next(csv.DictReader(f))
                    self.assertEqual(row["suite_type"], "milvus")
                    self.assertEqual(float(row["recall"]), 1)
                    report = (output / "milvus-fixture" / "report.md").read_text()
                    self.assertIn("Probes (nprobe)" if kind == "ivfflat" else "EF Search", report)
                    self.client = FakeClient()

    def test_preserves_explicit_and_positional_ids(self):
        batches = list(mv.embedding_batches(self.dataset(), 3))
        self.assertEqual([len(b) for b in batches], [3, 1])
        self.assertEqual([r["id"] for b in batches for r in b], self.ids)
        ds = {**self.dataset(), "train": self.train}
        self.assertEqual([r["id"] for b in mv.embedding_batches(ds, 2) for r in b], list(range(4)))

    def test_reuse_overwrite_and_index_verification(self):
        suite = self.suite()
        ds = self.dataset()
        name = mv.collection_name(self.config(), ds)
        suite.add_embeddings(self.client, name, ds)
        with self.assertRaisesRegex(ValueError, "already exists"):
            suite.add_embeddings(self.client, name, self.dataset())
        self.assertNotIn("drop_collection", self.client.events)
        suite.verify_collection(self.client, name, ds)
        suite.create_index(self.client, name, self.config(), ds)
        suite.verify_index(self.client, name, self.config(), ds)
        description = self.client.collections[name]["indexes"][mv.INDEX_NAME]
        # Server versions may return string parameters inside a nested params dict.
        description["params"] = {"nlist": str(description.pop("nlist"))}
        suite.verify_index(self.client, name, self.config(), ds)
        wrong = {**self.config(), "lists": 3}
        with self.assertRaisesRegex(ValueError, "nlist"):
            suite.verify_index(self.client, name, wrong, ds)
        self.client.collections[name]["indexes"][mv.INDEX_NAME]["state"] = "InProgress"
        with self.assertRaisesRegex(ValueError, "not fully built"):
            suite.verify_index(self.client, name, self.config(), ds)
        suite.overwrite_table = True
        suite.add_embeddings(self.client, name, self.dataset())
        self.assertIn("drop_collection", self.client.events)
        self.assertEqual(self.client.get_collection_stats(name)["row_count"], 4)

    def test_bad_vectors_incomplete_load_and_missing_index(self):
        ds = {**self.dataset(), "train": np.array([[np.nan, 0]])}
        with self.assertRaisesRegex(ValueError, "Invalid embeddings"):
            list(mv.embedding_batches(ds, 2))
        suite = self.suite()
        ds = {**self.dataset(), "num": 5}
        with self.assertRaisesRegex(ValueError, "expected 5"):
            suite.add_embeddings(self.client, "partial", ds)
        with self.assertRaisesRegex(ValueError, "expected 5"):
            suite.verify_collection(self.client, "partial", ds)
        with patch.object(self.client, "describe_index", return_value=None):
            with self.assertRaisesRegex(ValueError, "Missing index"):
                suite.verify_index(self.client, "partial", self.config(), ds)

    def test_build_only_and_skip_flags(self):
        suite = self.suite(build_only=True)
        with patch.object(suite, "create_connection", return_value=self.client), \
             patch.object(mv.datasets, "get_dataset", side_effect=lambda _: self.dataset()), \
             patch.object(mv, "ResultsManager"):
            suite.run()
        self.assertNotIn("load", self.client.events)
        self.assertNotIn("search", self.client.events)
        self.client.events.clear()
        suite = self.suite(skip_add_embeddings=True, skip_index_creation=True)
        with patch.object(suite, "create_connection", return_value=self.client), \
             patch.object(mv.datasets, "get_dataset", side_effect=lambda _: self.dataset()), \
             patch.object(mv, "ResultsManager"):
            suite.run()
        self.assertNotIn("insert", self.client.events)
        self.assertNotIn("create_index", self.client.events)
        self.assertIn("search", self.client.events)

    def test_validation_before_connection(self):
        for change in ({"lists": 0}, {"lists": True}, {"indexType": "flat"},
                       {"metric": "ip"}, {"vectorType": "halfvec"}, {"collection": "bad-name"},
                       {"top": 0}, {"benchmarks": {"bad": {"probes": 3}}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.suite({**self.config(), **change})
        with self.assertRaises(ValueError):
            self.suite(overwrite_table=True, skip_add_embeddings=True)
        with self.assertRaises(ValueError):
            self.suite({**self.config("hnsw"), "top": 11})

    def test_metrics_and_auto_lists(self):
        for metric, expected in (("l2", "L2"), ("cos", "COSINE"), ("ip", "IP")):
            self.assertEqual(mv.metric_type(metric), expected)
        self.assertEqual(mv.build_params({"indexType": "IVF_FLAT", "lists": "auto"}, {"num": 10**12}), {"nlist": 65536})
        self.assertEqual(mv.search_params(self.config("hnsw"), {"efSearch": 40}, self.metadata)["params"], {"ef": 40})

    def test_warmup_excluded_and_hdf5_closed(self):
        suite = self.suite()
        name = "test"
        suite.add_embeddings(self.client, name, self.dataset())
        suite.warmup = 3
        count = suite.warmup_queries(self.client, name, self.train.tolist(), 1,
                                    mv.search_params(self.config(), {"probes": 2}, self.metadata))
        self.assertEqual(count, 3)
        f = h5py.File(self.path / "train.hdf5", "w")
        train = f.create_dataset("train", data=self.train)
        mv.close_training_data(train)
        self.assertFalse(f.id.valid)

    def test_example_configs_and_cli(self):
        for path in Path("config").glob("*/milvus-*.yaml"):
            mv.TestSuite(str(path))
        args = mv.build_arg_parse().parse_args(["-s", "example.yaml"])
        self.assertEqual(args.url, "http://localhost:19530")
        self.assertNotIn("centroids_file", vars(args))
        self.assertNotIn("centroids-file", mv.build_arg_parse().format_help())

    def test_main_constructs_suite_from_cli(self):
        for path in ("config/cohere-1m-cos/milvus-ivfflat-1k.yaml",
                     "config/cohere-1m-cos/milvus-m16-128.yaml"):
            with self.subTest(path=path), \
                 patch("sys.argv", ["milvus_suite.py", "-s", path,
                                    "--url", "http://localhost:19530"]), \
                 patch.object(mv.TestSuite, "run", autospec=True) as run:
                mv.main()
                run.assert_called_once()
                suite = run.call_args.args[0]
                self.assertEqual(suite.url, "http://localhost:19530")
                self.assertEqual(next(iter(suite.config.values()))["dataset"], "cohere-1m-cos")


if __name__ == "__main__":
    unittest.main()
