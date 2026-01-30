import os
import requests
import h5py
import numpy as np
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---

DATA_DIR = "./datasets"
DATASETS = {
    # --- Standard HDF5 Datasets ---
    "laion-5m-test-ip": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/laion-5m-test-ip.hdf5",
        "metric": "ip",
        "type": "hdf5"
    },
    "laion-20m-test-ip": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/laion-20m-test-ip.hdf5",
        "metric": "ip",
        "type": "hdf5"
    },
    "laion-100m-test-ip": {
        "url": "https://enterprisedb-vector-datasets.s3.amazonaws.com/laion-100m-test-ip.hdf5",
        "metric": "ip",
        "type": "hdf5"
    },

    # --- Custom NPY Datasets ---
    "laion-400m-test-ip": {
        "type": "laion-multipart",
        "metric": "ip",
        "parts": 409,
        "dim": 512,
        "num": 400_000_000,
        "base_dir": os.path.join(DATA_DIR, "laion-400m/"),
        # Link to the file you just salvaged and uploaded
        "gt_url": "https://edb-vector-datasets.s3.amazonaws.com/laion/laion_400m_gt.npy",
        "gt_file": "laion_400m_gt.npy"
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
            "base": "https://edb-vector-datasets.s3.amazonaws.com/deep1b/deep1b_base.npy",
            "query": "https://edb-vector-datasets.s3.amazonaws.com/deep1b/deep1b_queries.npy",
            "groundtruth": "https://edb-vector-datasets.s3.amazonaws.com/deep1b/deep1b_groundtruth.npy"
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
    file_name = Path(info["url"]).name
    file_path = os.path.join(DATA_DIR, file_name)
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

    print(f"Loading Deep1B from {base_dir}...")

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

# --- FACTORY FUNCTION ---

def get_dataset(dataset_name):
    """Factory function to get a standardized dataset object."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    info = DATASETS[dataset_name]
    dtype = info.get("type", "hdf5")

    print(f"Loading dataset info for: {dataset_name} ({dtype})...")

    if dtype == "hdf5":
        return _load_hdf5_dataset(dataset_name, info)
    elif dtype == "laion-multipart":
        return _load_laion_multipart(dataset_name, info)
    elif dtype == "deep1b-mmap":
        return _load_deep1b_mmap(dataset_name, info)
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")