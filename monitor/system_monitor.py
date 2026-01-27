"""
System Monitor - Portable System Metrics Collection

Uses psutil for cross-platform system monitoring.
No dependency on iostat or other OS-specific tools.
"""

import os
import platform
import re
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd
import psutil


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Format a markdown table with proper column alignment."""
    num_cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                widths[i] = max(widths[i], len(str(cell)))

    lines = []
    header_cells = [h.ljust(widths[i]) for i, h in enumerate(headers)]
    lines.append("| " + " | ".join(header_cells) + " |")
    sep_cells = ["-" * widths[i] for i in range(num_cols)]
    lines.append("|-" + "-|-".join(sep_cells) + "-|")
    for row in rows:
        data_cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(data_cells) + " |")
    return lines


def is_local_database(url: str) -> bool:
    """
    Check if the database URL points to a local database.

    Returns True if the host is localhost, 127.0.0.1, ::1, or the local hostname.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"

        # Common localhost identifiers
        local_hosts = {"localhost", "127.0.0.1", "::1", ""}

        # Add the actual hostname and its aliases
        try:
            local_hosts.add(socket.gethostname().lower())
            local_hosts.add(socket.getfqdn().lower())
        except Exception:
            pass

        return host.lower() in local_hosts
    except Exception:
        # If parsing fails, assume local for safety
        return True


def get_pg_data_directory(conn) -> str:
    """Query PostgreSQL for its data directory."""
    with conn.cursor() as cur:
        cur.execute("SHOW data_directory")
        result = cur.fetchone()
        return result[0] if result else None


def get_block_device_for_path(path: str) -> str:
    """
    Determine the block device for a given filesystem path.

    Works on Linux and macOS. Returns the device name that should be monitored
    for IO stats. For LVM/RAID, returns the logical device (dm-X, md) to avoid
    counting IO multiple times from underlying physical devices.
    """
    if not os.path.exists(path):
        return None

    system = platform.system()

    try:
        if system == "Linux":
            # Use df to get the device
            result = subprocess.run(
                ["df", "--output=source", path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    device_path = lines[1].strip()  # e.g., /dev/nvme0n1p1 or /dev/mapper/vg-lv

                    # For LVM (/dev/mapper/*): resolve to dm-X device
                    # This is the correct device to monitor - NOT the underlying PVs
                    if "/dev/mapper/" in device_path:
                        try:
                            resolved = os.path.realpath(device_path)
                            return os.path.basename(resolved)  # Returns dm-X
                        except Exception:
                            return os.path.basename(device_path)

                    # For existing dm-X devices
                    if "/dev/dm-" in device_path:
                        return os.path.basename(device_path)

                    # For RAID (/dev/mdX): use the md device directly
                    if "/dev/md" in device_path:
                        return os.path.basename(device_path)

                    # For regular partitions: get the base device (for iostat compatibility)
                    device_name = os.path.basename(device_path)

                    # Handle nvme devices (nvme0n1p1 -> nvme0n1)
                    nvme_match = re.match(r"(nvme\d+n\d+)p?\d*", device_name)
                    if nvme_match:
                        return nvme_match.group(1)

                    # Handle standard devices (sda1 -> sda, vda1 -> vda)
                    std_match = re.match(r"([a-z]+)\d*", device_name)
                    if std_match:
                        return std_match.group(1)

                    return device_name

        elif system == "Darwin":  # macOS
            # Use df to get the device
            result = subprocess.run(
                ["df", path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    device_path = lines[1].split()[0]  # e.g., /dev/disk1s1
                    device_name = os.path.basename(device_path)
                    # Remove partition suffix (disk1s1 -> disk1)
                    match = re.match(r"(disk\d+)", device_name)
                    if match:
                        return match.group(1)
                    return device_name

    except Exception:
        pass

    return None


def detect_pg_io_device(conn) -> list[str]:
    """
    Auto-detect the block device(s) used by PostgreSQL.

    Returns a list of device names for IO monitoring.
    """
    data_dir = get_pg_data_directory(conn)
    if not data_dir:
        return None

    device = get_block_device_for_path(data_dir)
    if device:
        return [device]

    return None


class SystemMonitor:
    """
    Portable system monitoring using psutil.

    Collects CPU, memory, and disk IO metrics without relying on
    OS-specific tools like iostat. Supports continuous background
    monitoring with configurable sampling interval.
    """

    def __init__(self, results_dir: str = "./results", devices: Optional[list[str]] = None,
                 sample_interval: float = 1.0):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.devices = devices  # Specific devices to monitor (None = aggregate all)
        self.sample_interval = sample_interval  # Seconds between samples
        self.samples = []
        self.phase_markers = []  # [(timestamp, phase_name), ...]
        self._start_time = None
        self._prev_disk_io = None
        self._monitor_thread = None
        self._stop_event = None

    def start(self):
        """Start continuous background monitoring."""
        import threading
        if self._monitor_thread is not None:
            return  # Already running

        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        """Stop background monitoring."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            self.capture_sample()
            self._stop_event.wait(self.sample_interval)

    def _get_disk_io_counters(self):
        """
        Get disk IO counters for configured devices.

        If specific devices are configured, returns aggregated counters
        for those devices only. Otherwise returns system-wide counters.
        """
        if not self.devices:
            # No specific devices - use system-wide aggregate
            return psutil.disk_io_counters()

        # Get per-device counters and aggregate for our devices
        all_counters = psutil.disk_io_counters(perdisk=True)
        if not all_counters:
            return None

        # Aggregate counters for our specific devices
        read_count = 0
        write_count = 0
        read_bytes = 0
        write_bytes = 0
        found_devices = []

        for device in self.devices:
            if device in all_counters:
                counters = all_counters[device]
                read_count += counters.read_count
                write_count += counters.write_count
                read_bytes += counters.read_bytes
                write_bytes += counters.write_bytes
                found_devices.append(device)

        if not found_devices:
            # None of our devices found - fall back to aggregate
            # This likely means device name mismatch between detection and psutil
            print(f"Warning: Devices {self.devices} not found in psutil. "
                  f"Available: {list(all_counters.keys())[:10]}... Falling back to system-wide aggregate.")
            return psutil.disk_io_counters()

        # Return a named tuple-like object with the aggregated values
        class AggregatedDiskIO:
            pass
        result = AggregatedDiskIO()
        result.read_count = read_count
        result.write_count = write_count
        result.read_bytes = read_bytes
        result.write_bytes = write_bytes
        return result

    def _get_disk_io_rates(self) -> dict:
        """
        Get disk IO rates (delta since last call).

        Returns rates in ops/sec and bytes/sec.
        """
        current = self._get_disk_io_counters()
        current_time = time.time()

        if current is None:
            return {
                "read_iops": 0,
                "write_iops": 0,
                "read_bytes_sec": 0,
                "write_bytes_sec": 0,
            }

        if self._prev_disk_io is None:
            self._prev_disk_io = (current, current_time)
            return {
                "read_iops": 0,
                "write_iops": 0,
                "read_bytes_sec": 0,
                "write_bytes_sec": 0,
            }

        prev, prev_time = self._prev_disk_io
        elapsed = current_time - prev_time

        # Need at least 0.5 seconds for meaningful rate calculation
        if elapsed < 0.5:
            return {
                "read_iops": 0,
                "write_iops": 0,
                "read_bytes_sec": 0,
                "write_bytes_sec": 0,
            }

        rates = {
            "read_iops": (current.read_count - prev.read_count) / elapsed,
            "write_iops": (current.write_count - prev.write_count) / elapsed,
            "read_bytes_sec": (current.read_bytes - prev.read_bytes) / elapsed,
            "write_bytes_sec": (current.write_bytes - prev.write_bytes) / elapsed,
        }

        self._prev_disk_io = (current, current_time)
        return rates

    def capture_sample(self) -> dict:
        """Capture a single sample of system metrics."""
        if self._start_time is None:
            self._start_time = time.time()

        cpu_times = psutil.cpu_times_percent(interval=0)
        memory = psutil.virtual_memory()
        disk_io = self._get_disk_io_rates()

        sample = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": time.time() - self._start_time,
            # CPU metrics
            "cpu_percent": psutil.cpu_percent(),
            "cpu_user": cpu_times.user,
            "cpu_system": cpu_times.system,
            "cpu_iowait": getattr(cpu_times, "iowait", 0),  # Not available on all platforms
            # Memory metrics
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            # Disk IO metrics
            "disk_read_iops": disk_io["read_iops"],
            "disk_write_iops": disk_io["write_iops"],
            "disk_read_mb_sec": disk_io["read_bytes_sec"] / (1024**2),
            "disk_write_mb_sec": disk_io["write_bytes_sec"] / (1024**2),
        }

        self.samples.append(sample)
        return sample

    def mark_phase(self, phase_name: str):
        """Mark the start of a new phase for chart annotations."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        self.phase_markers.append((elapsed, phase_name))

    def capture_until_event(self, finish_event, interval: float = 1.0):
        """
        Capture samples until the finish event is set.

        Args:
            finish_event: threading.Event that signals when to stop
            interval: Seconds between samples
        """
        while not finish_event.is_set():
            self.capture_sample()
            time.sleep(interval)

        # Capture final sample
        self.capture_sample()

    def get_dataframe(self) -> pd.DataFrame:
        """Get samples as a pandas DataFrame."""
        return pd.DataFrame(self.samples)

    def save_csv(self, filename: str = "system_metrics.csv") -> Path:
        """Save samples to CSV file."""
        filepath = self.results_dir / filename
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
        return filepath

    def generate_dashboard(self, suite_name: str) -> Path:
        """
        Generate a combined dashboard chart with all metrics.

        Returns path to the generated PNG file.
        """
        df = self.get_dataframe()
        if df.empty:
            return None

        filepath = self.results_dir / f"{suite_name}_system_dashboard.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"System Metrics - {suite_name}", fontsize=14, fontweight="bold")

        elapsed = df["elapsed_sec"]

        # CPU Usage (top-left)
        ax1 = axes[0, 0]
        ax1.fill_between(elapsed, df["cpu_user"], alpha=0.5, label="User", color="#3498db")
        ax1.fill_between(elapsed, df["cpu_user"], df["cpu_user"] + df["cpu_system"],
                         alpha=0.5, label="System", color="#e74c3c")
        if "cpu_iowait" in df.columns and df["cpu_iowait"].sum() > 0:
            ax1.fill_between(elapsed, df["cpu_user"] + df["cpu_system"],
                             df["cpu_user"] + df["cpu_system"] + df["cpu_iowait"],
                             alpha=0.5, label="IO Wait", color="#f39c12")
        ax1.set_ylabel("CPU %")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_title("CPU Utilization")
        ax1.legend(loc="upper right")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        self._add_phase_markers(ax1)

        # Memory Usage (top-right)
        ax2 = axes[0, 1]
        ax2.plot(elapsed, df["memory_percent"], color="#9b59b6", linewidth=2)
        ax2.fill_between(elapsed, df["memory_percent"], alpha=0.3, color="#9b59b6")
        ax2.set_ylabel("Memory %")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_title("Memory Utilization")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        self._add_phase_markers(ax2)

        # Add secondary y-axis for absolute memory
        ax2_twin = ax2.twinx()
        ax2_twin.plot(elapsed, df["memory_used_gb"], color="#9b59b6", linestyle="--", alpha=0.5)
        ax2_twin.set_ylabel("Memory Used (GB)", color="#9b59b6", alpha=0.7)

        # Disk IOPS (bottom-left)
        ax3 = axes[1, 0]
        ax3.plot(elapsed, df["disk_read_iops"], label="Read IOPS", color="#2ecc71", linewidth=1.5)
        ax3.plot(elapsed, df["disk_write_iops"], label="Write IOPS", color="#e67e22", linewidth=1.5)
        ax3.set_ylabel("IOPS")
        ax3.set_xlabel("Time (seconds)")
        ax3.set_title("Disk IOPS")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
        self._add_phase_markers(ax3)

        # Disk Throughput (bottom-right)
        ax4 = axes[1, 1]
        ax4.plot(elapsed, df["disk_read_mb_sec"], label="Read MB/s", color="#2ecc71", linewidth=1.5)
        ax4.plot(elapsed, df["disk_write_mb_sec"], label="Write MB/s", color="#e67e22", linewidth=1.5)
        ax4.set_ylabel("Throughput (MB/s)")
        ax4.set_xlabel("Time (seconds)")
        ax4.set_title("Disk Throughput")
        ax4.legend(loc="upper right")
        ax4.grid(True, alpha=0.3)
        self._add_phase_markers(ax4)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        return filepath

    def _add_phase_markers(self, ax):
        """Add vertical lines for phase markers (only start markers, skip overlapping)."""
        # Only show _start markers, and use cleaner names
        phase_display = {
            "load_start": "Load",
            "index_start": "Index",
            "benchmark_start": "Benchmark",
        }
        colors = {"load_start": "#3498db", "index_start": "#e74c3c", "benchmark_start": "#2ecc71"}

        prev_elapsed = -999  # Track previous marker position
        for elapsed, phase in self.phase_markers:
            if phase not in phase_display:
                continue  # Skip _end markers

            # Skip if too close to previous marker (< 5 seconds apart)
            if elapsed - prev_elapsed < 5:
                continue

            color = colors.get(phase, "#9b59b6")
            ax.axvline(x=elapsed, color=color, linestyle="--", alpha=0.7, linewidth=1)
            ax.text(elapsed, ax.get_ylim()[1] * 0.95, f" {phase_display[phase]}",
                    rotation=90, va="top", ha="left", fontsize=8, color=color)
            prev_elapsed = elapsed

    def get_summary_stats(self) -> dict:
        """Get summary statistics for the monitoring period."""
        df = self.get_dataframe()
        if df.empty:
            return {}

        return {
            "duration_sec": df["elapsed_sec"].max(),
            "cpu_avg": df["cpu_percent"].mean(),
            "cpu_max": df["cpu_percent"].max(),
            "memory_avg_percent": df["memory_percent"].mean(),
            "memory_max_percent": df["memory_percent"].max(),
            "memory_max_gb": df["memory_used_gb"].max(),
            "disk_read_iops_avg": df["disk_read_iops"].mean(),
            "disk_write_iops_avg": df["disk_write_iops"].mean(),
            "disk_read_iops_max": df["disk_read_iops"].max(),
            "disk_write_iops_max": df["disk_write_iops"].max(),
            "disk_read_mb_sec_avg": df["disk_read_mb_sec"].mean(),
            "disk_write_mb_sec_avg": df["disk_write_mb_sec"].mean(),
            "disk_read_mb_sec_max": df["disk_read_mb_sec"].max(),
            "disk_write_mb_sec_max": df["disk_write_mb_sec"].max(),
        }

    def format_for_report(self) -> str:
        """Format summary stats as markdown for the report."""
        stats = self.get_summary_stats()
        if not stats:
            return ""

        lines = [
            "## System Metrics",
            "",
            f"**Monitoring Duration:** {stats['duration_sec']:.1f} seconds",
            "",
            "### CPU",
            "",
        ]
        lines.extend(format_markdown_table(
            ["Metric", "Value"],
            [
                ["Average", f"{stats['cpu_avg']:.1f}%"],
                ["Maximum", f"{stats['cpu_max']:.1f}%"],
            ]
        ))

        lines.extend([
            "",
            "### Memory",
            "",
        ])
        lines.extend(format_markdown_table(
            ["Metric", "Value"],
            [
                ["Average", f"{stats['memory_avg_percent']:.1f}%"],
                ["Maximum", f"{stats['memory_max_percent']:.1f}% ({stats['memory_max_gb']:.1f} GB)"],
            ]
        ))

        lines.extend([
            "",
            "### Disk IO",
            "",
        ])
        lines.extend(format_markdown_table(
            ["Metric", "Read", "Write"],
            [
                ["IOPS (avg)", f"{stats['disk_read_iops_avg']:.0f}", f"{stats['disk_write_iops_avg']:.0f}"],
                ["IOPS (max)", f"{stats['disk_read_iops_max']:.0f}", f"{stats['disk_write_iops_max']:.0f}"],
                ["Throughput avg (MB/s)", f"{stats['disk_read_mb_sec_avg']:.1f}", f"{stats['disk_write_mb_sec_avg']:.1f}"],
                ["Throughput max (MB/s)", f"{stats['disk_read_mb_sec_max']:.1f}", f"{stats['disk_write_mb_sec_max']:.1f}"],
            ]
        ))
        lines.append("")

        return "\n".join(lines)


def generate_system_report(results_dir: str) -> Path:
    """
    Generate a system information report.

    This is a portable version that uses psutil where possible
    and gracefully handles missing system commands.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    filepath = results_path / "system_report.txt"

    lines = []

    # Host info
    lines.append("=" * 60)
    lines.append("SYSTEM INFORMATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Hostname and network
    lines.append("\n[Host Information]")
    lines.append(f"Hostname: {socket.gethostname()}")
    try:
        lines.append(f"IP Address: {socket.gethostbyname(socket.gethostname())}")
    except Exception:
        lines.append("IP Address: Unable to determine")

    # OS info
    lines.append("\n[Operating System]")
    lines.append(f"System: {platform.system()}")
    lines.append(f"Release: {platform.release()}")
    lines.append(f"Version: {platform.version()}")
    lines.append(f"Machine: {platform.machine()}")
    lines.append(f"Processor: {platform.processor()}")

    # CPU info
    lines.append("\n[CPU Information]")
    lines.append(f"Physical cores: {psutil.cpu_count(logical=False)}")
    lines.append(f"Logical cores: {psutil.cpu_count(logical=True)}")

    try:
        freq = psutil.cpu_freq()
        if freq:
            lines.append(f"Max frequency: {freq.max:.0f} MHz")
            lines.append(f"Current frequency: {freq.current:.0f} MHz")
    except Exception:
        pass

    # Memory info
    lines.append("\n[Memory Information]")
    mem = psutil.virtual_memory()
    lines.append(f"Total: {mem.total / (1024**3):.2f} GB")
    lines.append(f"Available: {mem.available / (1024**3):.2f} GB")
    lines.append(f"Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")

    swap = psutil.swap_memory()
    lines.append(f"Swap Total: {swap.total / (1024**3):.2f} GB")
    lines.append(f"Swap Used: {swap.used / (1024**3):.2f} GB ({swap.percent}%)")

    # Disk info
    lines.append("\n[Disk Information]")
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            lines.append(f"\n  {partition.device}")
            lines.append(f"    Mountpoint: {partition.mountpoint}")
            lines.append(f"    Filesystem: {partition.fstype}")
            lines.append(f"    Total: {usage.total / (1024**3):.2f} GB")
            lines.append(f"    Used: {usage.used / (1024**3):.2f} GB ({usage.percent}%)")
            lines.append(f"    Free: {usage.free / (1024**3):.2f} GB")
        except (PermissionError, OSError):
            continue

    # Write report
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath
