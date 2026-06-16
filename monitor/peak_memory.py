"""Peak private-heap tracker for PostgreSQL index builds.

Samples ``RssAnon`` (private anonymous pages) from ``/proc/<pid>/status`` for
the leader backend plus every parallel worker, every ``interval_s`` seconds.
``RssAnon`` excludes shared-memory regions (``shared_buffers``) by kernel
definition, so summing across N backends does not double-count SB. Works
uniformly whether the suite uses parallel workers or not — non-parallel builds
just see an empty child list.

Linux-only (relies on ``/proc/<pid>/status``). Local-PG only (skipped for
remote URLs).
"""
import os
import threading

import psutil
import psycopg


def _read_rss_anon(pid: int) -> int | None:
    """Return RssAnon in bytes from /proc/<pid>/status, or None if unavailable.

    RssAnon is the private anonymous resident set — heap, mmap MAP_PRIVATE|
    MAP_ANON, stack. Excludes RssFile (binaries/libs) and RssShmem
    (shared_buffers, parallel-worker DSM segments).
    """
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    # Format: "RssAnon:\t   12345 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    return None


class PgBuildMemoryTracker:
    """Context manager that samples PG backend RssAnon while a build runs.

    Usage:
        with PgBuildMemoryTracker(conn, url) as tracker:
            conn.execute("CREATE INDEX ...")
        peak_bytes = tracker.peak_bytes  # None if remote / unsupported

    Reports the peak aggregate (leader + all parallel workers) seen across
    samples, plus per-component peaks. Aggregate is computed per-sample then
    maxed — so it reflects the highest instantaneous build footprint, not the
    sum of independent per-process maxima.
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        url: str,
        interval_s: float = 0.2,
    ):
        self.conn = conn
        self.url = url
        self.interval_s = interval_s

        self.peak_bytes: int | None = None
        self.leader_peak_bytes: int | None = None
        self.workers_peak_bytes: int | None = None
        self.peak_worker_count: int = 0
        self.samples: int = 0
        self.reason_skipped: str | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._leader_pid: int | None = None

    def __enter__(self):
        from monitor.system_monitor import is_local_database
        if not is_local_database(self.url):
            self.reason_skipped = "remote-pg"
            return self

        if not os.path.isdir("/proc"):
            self.reason_skipped = "no-procfs"
            return self

        try:
            row = self.conn.execute("SELECT pg_backend_pid()").fetchone()
            self._leader_pid = int(row[0])
        except (psycopg.Error, ValueError) as e:
            self.reason_skipped = f"pid-lookup: {e}"
            return self

        # Confirm we can actually read this PID's status (PG runs as the
        # `postgres` user; vsbt may run as a different user without read
        # access to /proc/<pid>/status of other users).
        if _read_rss_anon(self._leader_pid) is None:
            self.reason_skipped = f"procfs-unreadable for pid {self._leader_pid}"
            return self

        self.peak_bytes = 0
        self.leader_peak_bytes = 0
        self.workers_peak_bytes = 0
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        return False

    def _sample_loop(self):
        try:
            leader = psutil.Process(self._leader_pid)
        except psutil.NoSuchProcess:
            return

        while not self._stop.is_set():
            leader_anon = _read_rss_anon(self._leader_pid) or 0
            workers_anon = 0
            worker_count = 0
            try:
                for child in leader.children(recursive=True):
                    r = _read_rss_anon(child.pid)
                    if r is None:
                        continue
                    workers_anon += r
                    worker_count += 1
            except psutil.NoSuchProcess:
                break
            except psutil.Error:
                pass

            total = leader_anon + workers_anon
            if total > (self.peak_bytes or 0):
                self.peak_bytes = total
            if leader_anon > (self.leader_peak_bytes or 0):
                self.leader_peak_bytes = leader_anon
            if workers_anon > (self.workers_peak_bytes or 0):
                self.workers_peak_bytes = workers_anon
            if worker_count > self.peak_worker_count:
                self.peak_worker_count = worker_count
            self.samples += 1
            self._stop.wait(self.interval_s)


def format_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    gb = n / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    mb = n / (1024 ** 2)
    return f"{mb:.0f} MB"
