"""
loader.py — fast bulk vector ingestion via PostgreSQL COPY BINARY.

The sliceable path (HDF5, mmap) avoids Python-level per-row overhead by
packing an entire chunk into a COPY BINARY buffer with numpy before
sending. The generator path falls back to per-row write_row() since
generators do not allow random access.

Binary format per row (big-endian throughout):
  int16  field_count = 2
  int32  id_field_len = 4
  int32  id_value
  int32  vec_field_len = 4 + dim * 4
  int16  dim
  int16  unused = 0
  float32 × dim
"""

import struct
import threading
import time

import numpy as np
from tqdm import tqdm


_PGCOPY_HEADER = b"PGCOPY\n\377\r\n\0" + struct.pack(">II", 0, 0)  # 19 bytes
_PGCOPY_TRAILER = struct.pack(">h", -1)  # 2 bytes


def pack_copy_binary_chunk(ids: np.ndarray, vecs: np.ndarray) -> bytes:
    """
    Pack (ids, vecs) into a self-contained COPY BINARY buffer.

    ids:  int32 array of shape (N,)
    vecs: float32 array of shape (N, D)

    Returns bytes ready to pass to copy.write() for a new COPY BINARY
    command. No Python loop; all packing is done with numpy.
    """
    N, D = vecs.shape
    vec_field_len = 4 + D * 4  # int16 dim + int16 unused + float32×D

    # Fixed per-row header — 14 bytes, big-endian structured array
    fixed = np.empty(N, dtype=[
        ("field_count", ">i2"),   # 2 bytes: always 2
        ("id_field_len", ">i4"),  # 4 bytes: always 4
        ("id_val",       ">i4"),  # 4 bytes: the id
        ("vec_field_len", ">i4"), # 4 bytes: 4 + D*4
    ])
    fixed["field_count"]  = 2
    fixed["id_field_len"] = 4
    fixed["id_val"]       = ids
    fixed["vec_field_len"] = vec_field_len

    # dim prefix — 4 bytes per row
    dim_prefix = np.zeros(N, dtype=[("dim", ">i2"), ("unused", ">i2")])
    dim_prefix["dim"] = D

    # Vector values — big-endian float32
    vec_be = vecs if vecs.dtype == np.dtype(">f4") else vecs.astype(">f4")

    # Assemble: concatenate the byte views along axis=1
    fixed_b  = fixed.view(np.uint8).reshape(N, 14)
    dim_b    = dim_prefix.view(np.uint8).reshape(N, 4)
    vec_b    = vec_be.view(np.uint8).reshape(N, D * 4)

    row_data = np.concatenate([fixed_b, dim_b, vec_b], axis=1)  # (N, 18 + D*4)
    return _PGCOPY_HEADER + row_data.tobytes() + _PGCOPY_TRAILER


def load_vectors(
    conn_factory,
    table_name: str,
    data,           # sliceable (HDF5 / mmap) — must support data[i:j]
    n: int,
    chunk_size: int = 500_000,
    num_threads: int = 4,
    progress: tqdm = None,
) -> None:
    """
    Load vectors from a sliceable source into *table_name* using the fast
    numpy binary-pack path. Each chunk is sent as a single COPY BINARY
    payload, avoiding per-row Python overhead.

    conn_factory: callable returning a psycopg Connection (autocommit)
    data:         sliceable — data[i:j] returns an (N, D) numpy array
    progress:     optional tqdm instance — updated with the number of rows
                  loaded per batch
    """
    def _load_chunk(chunk_start: int, chunk_len: int) -> None:
        conn = conn_factory()
        chunk = data[chunk_start: chunk_start + chunk_len]
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        ids = np.arange(chunk_start, chunk_start + len(chunk), dtype=np.int32)
        buf = pack_copy_binary_chunk(ids, chunk)

        with conn.cursor().copy(
            f"COPY {table_name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.write(buf)
            while conn.pgconn.flush() == 1:
                time.sleep(0)
        conn.close()

    if num_threads > 1:
        threads: list[threading.Thread] = []
        batch_rows = 0

        for i in range(0, n, chunk_size):
            chunk_len = min(chunk_size, n - i)
            t = threading.Thread(target=_load_chunk, args=(i, chunk_len))
            threads.append(t)
            batch_rows += chunk_len

            if len(threads) >= num_threads or (i + chunk_len) >= n:
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                if progress is not None:
                    progress.update(batch_rows)
                threads = []
                batch_rows = 0
    else:
        for i in range(0, n, chunk_size):
            chunk_len = min(chunk_size, n - i)
            _load_chunk(i, chunk_len)
            if progress is not None:
                progress.update(chunk_len)


def load_vectors_generator(
    conn,
    table_name: str,
    data,           # iterable yielding (id, vector) pairs
    n: int,
    progress: tqdm = None,
) -> None:
    """
    Load vectors from a generator source (e.g. LAION multi-part NPY)
    using per-row COPY BINARY. No random access available so the fast
    numpy path cannot be used here.
    """
    with conn.cursor().copy(
        f"COPY {table_name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["integer", "vector"])
        for i, vec in data:
            copy.write_row((i, vec))
            while conn.pgconn.flush() == 1:
                time.sleep(0)
            if progress is not None:
                progress.update(1)
