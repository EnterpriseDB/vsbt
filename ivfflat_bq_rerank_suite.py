"""
IVFFlat BQ Rerank Benchmark Suite

Entry point for benchmarking IVFFlat with binary quantization and
halfvec re-ranking. Delegates to pgvector_suite.IVFFlatBQRerankTestSuite.
"""

import argparse

import common
from pgvector_suite import IVFFlatBQRerankTestSuite


def build_arg_parse():
    """Build argument parser for IVFFlat BQ Rerank benchmark suite."""
    parser = argparse.ArgumentParser(description="IVFFlat BQ Rerank Benchmark Suite")
    common.build_arg_parse(parser)
    return parser


def main():
    """Main entry point for IVFFlat BQ Rerank benchmark suite."""
    parser = build_arg_parse()
    args = parser.parse_args()

    test_suite = IVFFlatBQRerankTestSuite(
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
    )

    test_suite.run()
    print("Test suite completed.")


if __name__ == "__main__":
    main()
