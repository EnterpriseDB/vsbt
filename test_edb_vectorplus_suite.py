"""Self-check for edb_vectorplus_suite's pure helpers. No DB, no framework.

    python test_edb_vectorplus_suite.py
"""

import edb_vectorplus_suite as vp


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def demo():
    # --- metric tables ---
    assert vp.metric_opclass("cos") == "vector_cosine_ops"
    assert vp.metric_opclass("euclidean") == "vector_l2_ops"
    assert vp.metric_opclass("cos", "halfvec") == "halfvec_cosine_ops"
    assert vp.metric_opclass("ip", "halfvec") == "halfvec_ip_ops"
    assert vp.metric_operator("ip") == "<#>"
    assert _raises(vp.metric_opclass, "jaccard")
    assert _raises(vp.metric_operator, "jaccard")

    # --- lists resolution ---
    assert vp.resolve_lists({"lists": 2236}, {"num": 5_000_000}) == 2236
    assert vp.resolve_lists({"lists": "auto"}, {"num": 1_000_000}) == 1000
    assert vp.resolve_lists({"lists": "AUTO"}, {"num": 4}) == 2
    assert _raises(vp.resolve_lists, {"lists": 0}, {"num": 10})
    assert _raises(vp.resolve_lists, {"lists": "none"}, {"num": 10})

    # --- session GUCs ---
    assert vp.probe_gucs({"probes": 80})[0] == "SET ivfplus.probes = 80"
    assert "SET enable_seqscan = off" in vp.probe_gucs({"probes": 1})

    # --- DDL ---
    ddl = vp.create_index_sql(
        "cohere_1m_cos", {"lists": 1000}, {"metric": "cos", "num": 1_000_000}
    )
    assert "CREATE INDEX cohere_1m_cos_embedding_idx ON cohere_1m_cos" in ddl
    assert "USING ivfplus (embedding vector_cosine_ops)" in ddl
    assert ddl.endswith("WITH (lists = 1000)")
    ddl = vp.create_index_sql(
        "t_halfvec", {"lists": 10},
        {"metric": "ip", "num": 100, "vector_type": "halfvec"},
    )
    assert "ON t_halfvec USING ivfplus (embedding halfvec_ip_ops)" in ddl

    # --- query template matches common.TestSuite.warmup_query ---
    import common
    sql, bind = common.TestSuite.warmup_query(
        None, "tbl", {}, "<=>", 10, {"probes": 20}
    )
    assert sql == vp.search_query_sql("tbl", "<=>", 10)
    assert bind("q") == ("q",)

    # --- report column specs ---
    label, extract = vp.CONFIG_COLUMNS[0]
    assert label == "Lists"
    assert extract({"lists": 1000}, {}) == "1000"
    assert extract({}, {"lists": 2236}) == "2236"
    label, extract = vp.CONFIG_COLUMNS[1]
    assert label == "Vector Type"
    assert extract({}, {}) == "vector"
    assert extract({"vectorType": "halfvec"}, {}) == "halfvec"
    assert vp.BENCH_COLUMNS == (("probes", "Probes"),)

    # --- vectorType validated at construction, before any connection ---
    import tempfile

    def _suite(vector_type):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(f"t:\n  indexType: ivfplus\n  vectorType: {vector_type}\n"
                    f"  dataset: openai-5k-cos\n  metric: cos\n  lists: 1\n"
                    f"  top: 1\n  benchmarks: {{}}\n")
        return vp.TestSuite(suite_file=f.name,
                            url="postgresql://x@localhost:5432/postgres",
                            devices=None, chunk_size=1)

    _suite("halfvec")
    assert _raises(_suite, "bit")


if __name__ == "__main__":
    demo()
    print("ok")
