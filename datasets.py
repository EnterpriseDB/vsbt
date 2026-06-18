import os
import glob
import requests
import h5py
import numpy as np
import pyarrow.parquet as pq
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = os.environ.get("DATASET_LOCAL_DIR", "./datasets")
DATASETS = {
    # --- Standard HDF5 Datasets ---
    "laion-5m-test-ip": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/laion/5m/base.hdf5",
        "metric": "ip",
        "type": "hdf5",
        "dim": 768,
        "num": 5_000_000
    },
    "laion-20m-test-ip": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/laion/20m/base.hdf5",
        "metric": "ip",
        "type": "hdf5",
        "dim": 768,
        "num": 20_000_000
    },
    "laion-100m-test-ip": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/laion/100m/base.hdf5",
        "metric": "ip",
        "type": "hdf5",
        "dim": 768,
        "num": 100_000_000
    },
    "sift-128-euclidean": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/sift/1m/base.hdf5",
        "metric": "l2",
        "type": "hdf5",
        "dim": 128,
        "num": 1_000_000
    },
    "glove-100-angular": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/glove/1m/base.hdf5",
        "metric": "cos",
        "type": "hdf5",
        "dim": 100,
        "num": 1_183_514
    },
    "gist-960-euclidean": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/gist/1m/base.hdf5",
        "metric": "l2",
        "type": "hdf5",
        "dim": 960,
        "num": 1_000_000
    },
    "dbpedia-openai-1000k-angular": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/dbpedia/1m/base.hdf5",
        "metric": "cos",
        "type": "hdf5",
        "dim": 1536,
        "num": 990_000
    },

    # --- Custom NPY Datasets ---
    "laion-400m-test-ip": {
        "type": "laion-multipart",
        "metric": "ip",
        "parts": 409,
        "dim": 512,
        "num": 400_000_000,
        "base_dir": os.path.join(DATA_DIR, "laion/400m/parts/"),
        "gt_url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/laion/400m/gt.npy",
        "gt_file": "laion_400m_gt.npy"
    },

    # --- Parquet Datasets (OpenAI, Cohere) ---
    "openai-500k-cos": {
        "type": "parquet",
        "metric": "cos",
        "dim": 1536,
        "num": 500_000,
        "s3_prefix": "s3://enterprisedb-vector-datasets/openai/500k",
        "base_dir": os.path.join(DATA_DIR, "openai/500k"),
    },
    "openai-1m-cos": {
        "type": "parquet",
        "metric": "cos",
        "dim": 1536,
        "num": 1_000_000,
        "s3_prefix": "s3://enterprisedb-vector-datasets/openai/1m",
        "base_dir": os.path.join(DATA_DIR, "openai/1m"),
    },
    "openai-2m-cos": {
        "type": "parquet",
        "metric": "cos",
        "dim": 1536,
        "num": 2_000_000,
        "s3_prefix": "s3://enterprisedb-vector-datasets/openai/2m",
        "base_dir": os.path.join(DATA_DIR, "openai/2m"),
    },
    "openai-5m-cos": {
        "type": "parquet",
        "metric": "cos",
        "dim": 1536,
        "num": 5_000_000,
        "s3_prefix": "s3://enterprisedb-vector-datasets/openai/5m",
        "base_dir": os.path.join(DATA_DIR, "openai/5m"),
    },
    "cohere-1m-cos": {
        "type": "parquet",
        "metric": "cos",
        "dim": 768,
        "num": 1_000_000,
        "s3_prefix": "s3://enterprisedb-vector-datasets/cohere/1m",
        "base_dir": os.path.join(DATA_DIR, "cohere/1m"),
    },
    "cohere-2m-cos": {
        "type": "parquet",
        "metric": "cos",
        "dim": 768,
        "num": 2_000_000,
        "s3_prefix": "s3://enterprisedb-vector-datasets/cohere/2m",
        "base_dir": os.path.join(DATA_DIR, "cohere/2m"),
    },
    "cohere-3m-cos": {
        "type": "parquet",
        "metric": "cos",
        "dim": 768,
        "num": 3_000_000,
        "s3_prefix": "s3://enterprisedb-vector-datasets/cohere/3m",
        "base_dir": os.path.join(DATA_DIR, "cohere/3m"),
    },
    "cohere-10m-cos": {
        "type": "parquet",
        "metric": "cos",
        "dim": 768,
        "num": 10_000_000,
        "s3_prefix": "s3://enterprisedb-vector-datasets/cohere/10m",
        "base_dir": os.path.join(DATA_DIR, "cohere/10m"),
    },

    # --- Deep1B Configuration ---
    "deep1b-test-l2": {
        "type": "deep1b-mmap",
        "metric": "l2",
        "dim": 96,
        "num": 1_000_000_000,
        "base_dir": os.path.join(DATA_DIR, "deep1b"),
        # Direct URLs to your pre-converted NPY files and the IBIN ground truth
        "urls": {
            "base": "https://enterprisedb-vector-datasets.s3.amazonaws.com/deep1b/1b/base.npy",
            "query": "https://enterprisedb-vector-datasets.s3.amazonaws.com/deep1b/1b/queries.npy",
            "groundtruth": "https://enterprisedb-vector-datasets.s3.amazonaws.com/deep1b/1b/groundtruth.npy"
        },
        # Local filenames to save them as
        "files": {
            "base": "deep1b_base.npy",
            "query": "deep1b_queries.npy",
            "groundtruth": "deep1b_groundtruth.npy"
        }
    }
}


# --- DOWNLOAD UTILITIES ---

def download_http_file(url: str, path: str):
    """Robust downloader that downloads to .tmp first."""
    if os.path.exists(path):
        return

    print(f"Downloading {url} to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Download to a temp file first
    tmp_path = path + ".tmp"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for 403/404 BEFORE opening file

        total_size = int(response.headers.get('content-length', 0))

        with open(tmp_path, "wb") as f, tqdm(
                desc=os.path.basename(path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                ncols=80,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]',
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

        # Rename tmp to final only on success
        os.rename(tmp_path, path)

    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # Clean up garbage
        raise e


def _get_laion_url(part: int) -> str:
    return f"https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_{part}.npy"


def download_laion_parts(limit=409, max_workers=None):
    """Downloads LAION NPY parts in parallel."""
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    base_dir = DATASETS["laion-400m-test-ip"]["base_dir"]
    os.makedirs(base_dir, exist_ok=True)

    def _do_dl(idx):
        url = _get_laion_url(idx)
        path = os.path.join(base_dir, f"img_emb_{idx}.npy")
        download_http_file(url, path)

    print(f"Downloading LAION parts 0 to {limit}...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_do_dl, idx): idx for idx in range(limit + 1)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Failed part {futures[future]}: {e}")


# --- DATASET LOADERS ---

def _load_hdf5_dataset(name, info):
    """Handles standard HDF5 benchmark datasets."""
    file_path = os.path.join(DATA_DIR, f"{name}.hdf5")
    download_http_file(info["url"], file_path)

    f = h5py.File(file_path, "r")
    dim = int(f.attrs["dimension"]) if "dimension" in f.attrs else f["train"].shape[1]
    num = f["train"].shape[0]

    return {
        "name": name,
        "type": "hdf5",
        "metric": info["metric"],
        "dim": dim,
        "num": num,
        "train": f["train"],
        "test": f["test"][:],
        "neighbors": f["neighbors"][:]
    }


def _load_laion_multipart(name, info):
    """Handles the split NPY files for LAION-400M."""
    base_dir = info["base_dir"]
    parts = info["parts"]

    # 1. Download Ground Truth from S3
    gt_path = os.path.join(base_dir, info["gt_file"])
    if not os.path.exists(gt_path):
        print(f"Downloading LAION Ground Truth to {gt_path}...")
        download_http_file(info["gt_url"], gt_path)

    gt_data = np.load(gt_path)

    # 2. Generator for sequential reading of multiple files
    def laion_generator():
        uploaded_count = 0
        for idx in range(parts + 1):
            path = os.path.join(base_dir, f"img_emb_{idx}.npy")
            if not os.path.exists(path):
                continue
            data = np.load(path)
            for row in data:
                yield uploaded_count, row.reshape(-1)
                uploaded_count += 1
            del data

            # 3. Load Queries (from part 0)

    query_path = os.path.join(base_dir, "img_emb_0.npy")
    if os.path.exists(query_path):
        queries = np.load(query_path)[:100, :]
    else:
        # Fallback if download skipped
        queries = np.zeros((100, info["dim"]))

    return {
        "name": name,
        "type": "laion-multipart",
        "metric": info["metric"],
        "dim": info["dim"],
        "num": info["num"],
        "train": laion_generator(),
        "test": queries,
        "neighbors": gt_data
    }


def _load_deep1b_mmap(name, info):
    """
    Handles Deep1B.
    Checks if local .npy files exist. If not, downloads them from config URLs.
    """
    base_dir = info["base_dir"]
    files = info["files"]
    urls = info["urls"]

    path_base = os.path.join(base_dir, files["base"])
    path_query = os.path.join(base_dir, files["query"])
    path_gt = os.path.join(base_dir, files["groundtruth"])

    # 1. Download missing components
    if not os.path.exists(path_base):
        print(f"Deep1B Base not found locally. Downloading from {urls['base']}...")
        download_http_file(urls["base"], path_base)

    if not os.path.exists(path_query):
        print(f"Deep1B Queries not found locally. Downloading from {urls['query']}...")
        download_http_file(urls["query"], path_query)

    if not os.path.exists(path_gt):
        print(f"Deep1B GroundTruth not found locally. Downloading from {urls['groundtruth']}...")
        download_http_file(urls["groundtruth"], path_gt)

    # 2. Strict Check (in case download failed or URL was bad)
    missing = []
    if not os.path.exists(path_base): missing.append("Base")
    if not os.path.exists(path_query): missing.append("Query")
    if not os.path.exists(path_gt): missing.append("GroundTruth")

    if missing:
        raise FileNotFoundError(f"Deep1B Missing Files: {', '.join(missing)}")

    # 3. Load
    # allow_pickle=True is used assuming these are trusted local files
    train_data = np.load(path_base, mmap_mode='r', allow_pickle=True)
    test_data = np.load(path_query, allow_pickle=True)
    neighbors_data = np.load(path_gt, allow_pickle=True)

    return {
        "name": name,
        "type": "deep1b-mmap",
        "metric": info["metric"],
        "dim": info["dim"],
        "num": info["num"],
        "train": train_data,
        "test": test_data,
        "neighbors": neighbors_data
    }

def _s3_prefix_to_http_url(s3_prefix, filename):
    """Convert s3://bucket/key to https://bucket.s3.amazonaws.com/key/filename."""
    without_scheme = s3_prefix[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}/{filename}"


def _expected_parquet_files(num):
    """Return the list of expected parquet filenames for a dataset of size `num`."""
    files = ["test.parquet", "neighbors.parquet"]
    file_count = max(1, num // 1_000_000)
    if file_count == 1:
        files.append("shuffle_train.parquet")
    else:
        for i in range(file_count):
            files.append(f"shuffle_train-{i:02d}-of-{file_count:02d}.parquet")
    return files


def _download_parquet_from_s3(s3_prefix, base_dir, num):
    """Download parquet dataset files from S3 if not already present."""
    import shutil
    import subprocess

    os.makedirs(base_dir, exist_ok=True)

    existing = set(os.listdir(base_dir))
    needed = ["test.parquet", "neighbors.parquet"]
    has_train = any(f.startswith("shuffle_train") or f == "train.parquet" for f in existing)
    has_needed = all(f in existing for f in needed)

    if has_train and has_needed:
        return

    if shutil.which("aws"):
        print(f"Downloading dataset from {s3_prefix} ...")
        try:
            subprocess.run(
                ["aws", "s3", "sync", s3_prefix, base_dir,
                 "--exclude", "scalar_labels*",
                 "--exclude", "neighbors_*"],
                check=True,
            )
            return
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"Warning: aws s3 sync failed ({e}), falling back to HTTP...")
    else:
        print(f"Warning: aws CLI not found, downloading via HTTP (install aws CLI for faster parallel downloads)...")

    for filename in _expected_parquet_files(num):
        dest = os.path.join(base_dir, filename)
        if os.path.exists(dest):
            continue
        url = _s3_prefix_to_http_url(s3_prefix, filename)
        download_http_file(url, dest)


def _load_parquet(name, info):
    """Handles parquet datasets with train/test/neighbors files."""
    base_dir = info["base_dir"]

    if info.get("s3_prefix") and not os.path.exists(os.path.join(base_dir, "test.parquet")):
        _download_parquet_from_s3(info["s3_prefix"], base_dir, info["num"])

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    train_files = sorted(glob.glob(os.path.join(base_dir, "shuffle_train-*.parquet")))
    if not train_files:
        train_files = sorted(glob.glob(os.path.join(base_dir, "shuffle_train.parquet")))
    if not train_files:
        train_files = sorted(glob.glob(os.path.join(base_dir, "train.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train parquet files found in {base_dir}")

    test_path = os.path.join(base_dir, "test.parquet")
    neighbors_path = os.path.join(base_dir, "neighbors.parquet")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.parquet not found in {base_dir}")
    if not os.path.exists(neighbors_path):
        raise FileNotFoundError(f"neighbors.parquet not found in {base_dir}")

    test_table = pq.read_table(test_path)
    test_embs = np.array(test_table.column("emb").to_pylist(), dtype=np.float32)

    gt_table = pq.read_table(neighbors_path)
    neighbors = np.array(gt_table.column("neighbors_id").to_pylist())

    dim = info["dim"]
    num = info["num"]

    has_id_col = "id" in pq.ParquetFile(train_files[0]).schema.names

    def parquet_generator():
        row_id = 0
        for f in train_files:
            pf = pq.ParquetFile(f, memory_map=True)
            if has_id_col:
                for batch in pf.iter_batches(batch_size=10_000, columns=["id", "emb"]):
                    for rid, emb in zip(batch.column("id"), batch.column("emb")):
                        yield rid.as_py(), np.array(emb.as_py(), dtype=np.float32)
            else:
                for batch in pf.iter_batches(batch_size=10_000, columns=["emb"]):
                    for emb in batch.column("emb"):
                        yield row_id, np.array(emb.as_py(), dtype=np.float32)
                        row_id += 1

    return {
        "name": name,
        "type": "parquet",
        "metric": info["metric"],
        "dim": dim,
        "num": num,
        "train": parquet_generator(),
        "test": test_embs,
        "neighbors": neighbors,
    }


# --- FACTORY FUNCTION ---

def get_dataset(dataset_name):
    """Factory function to get a standardized dataset object."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    info = DATASETS[dataset_name]
    dtype = info.get("type", "hdf5")


    if dtype == "hdf5":
        return _load_hdf5_dataset(dataset_name, info)
    elif dtype == "laion-multipart":
        return _load_laion_multipart(dataset_name, info)
    elif dtype == "deep1b-mmap":
        return _load_deep1b_mmap(dataset_name, info)
    elif dtype == "parquet":
        return _load_parquet(dataset_name, info)
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")


# --- PUBLIC METADATA API ---

# Map internal loader types to public format names (matches `datasetType` in suite YAMLs).
_FORMAT = {
    "hdf5": "hdf5",
    "laion-multipart": "npy",
    "deep1b-mmap": "npy",
}


def list_datasets():
    """Return the public metadata view — name, metric, format, dim, num.

    Projects the internal DATASETS dict to a stable contract for external
    consumers (e.g. kube-pgperf UI). Loader-internal fields (url, base_dir,
    parts, gt_url, files) are intentionally excluded. The `format` field is
    the on-disk format (hdf5/npy) — matches `datasetType` in suite YAMLs —
    not the internal loader name.
    """
    out = []
    for name, info in DATASETS.items():
        entry = {"name": name, "metric": info["metric"]}
        loader_type = info.get("type", "hdf5")
        entry["format"] = _FORMAT.get(loader_type, loader_type)
        if "dim" in info:
            entry["dim"] = info["dim"]
        if "num" in info:
            entry["num"] = info["num"]
        out.append(entry)
    return out


def _cli():
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(description="vsbt dataset metadata")
    p.add_argument("--list", action="store_true", help="list available datasets")
    p.add_argument("--json", action="store_true", help="emit JSON (default: table)")
    args = p.parse_args()

    if not args.list:
        p.print_help()
        sys.exit(0)

    entries = list_datasets()
    if args.json:
        json.dump(entries, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    header = f"{'name':28} {'metric':6} {'format':6} {'dim':>5} {'num':>14}"
    print(header)
    print("-" * len(header))
    for e in entries:
        print(f"{e['name']:28} {e.get('metric', '-'):6} {e.get('format', '-'):6} "
              f"{e.get('dim', '-'):>5} {e.get('num', '-'):>14,}")


if __name__ == "__main__":
    _cli()