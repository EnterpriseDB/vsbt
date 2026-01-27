"""
PostgreSQL Statistics Collector

Collects PostgreSQL performance statistics at phase boundaries during benchmarks.
Only uses stats that don't require ANALYZE to be accurate.
"""

from datetime import datetime
from typing import Optional


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


class PGStatsCollector:
    """
    Collect PostgreSQL stats at benchmark phase boundaries.

    All stats collected here are live counters that don't depend on
    ANALYZE or VACUUM having been run.
    """

    def __init__(self, conn):
        """Initialize with a database connection."""
        self.conn = conn
        self.snapshots = {}
        self._pg_stat_statements_available = None
        # Capture custom settings immediately (connection may close later)
        self._custom_settings = self._fetch_custom_settings()

    # Settings to exclude from custom settings report (not relevant to benchmarks)
    SETTINGS_BLACKLIST = {
        'TimeZone',
        'default_text_search_config',
        'lc_messages',
        'lc_monetary',
        'lc_numeric',
        'lc_time',
        'log_timezone',
        'application_name',
        'client_encoding',
        'DateStyle',
        'IntervalStyle',
    }

    def _fetch_custom_settings(self) -> list[dict]:
        """
        Fetch PostgreSQL settings that differ from their default values.

        Called at initialization to capture settings before connection closes.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        name,
                        setting,
                        unit,
                        boot_val,
                        source,
                        category
                    FROM pg_settings
                    WHERE source NOT IN ('default', 'override')
                      AND setting != boot_val
                    ORDER BY category, name
                """)

                settings = []
                for row in cur.fetchall():
                    name, setting, unit, boot_val, source, category = row
                    # Skip blacklisted settings
                    if name in self.SETTINGS_BLACKLIST:
                        continue
                    settings.append({
                        "name": name,
                        "setting": setting,
                        "unit": unit or "",
                        "boot_val": boot_val,
                        "source": source,
                        "category": category,
                    })
                return settings
        except Exception:
            return []

    def get_custom_settings(self) -> list[dict]:
        """Get PostgreSQL settings that differ from their default values."""
        return self._custom_settings or []

    def format_custom_settings(self) -> str:
        """Format custom PostgreSQL settings as markdown."""
        settings = self.get_custom_settings()
        if not settings:
            return ""

        rows = []
        for s in settings:
            value = f"{s['setting']}{s['unit']}" if s['unit'] else s['setting']
            default = f"{s['boot_val']}{s['unit']}" if s['unit'] else s['boot_val']
            rows.append([s['name'], value, default, s['source']])

        lines = [
            "## PostgreSQL Configuration",
            "",
            "Settings modified from defaults:",
            "",
        ]
        lines.extend(format_markdown_table(["Setting", "Value", "Default", "Source"], rows))
        lines.append("")
        return "\n".join(lines)

    def _check_pg_stat_statements(self) -> bool:
        """Check if pg_stat_statements extension is available."""
        if self._pg_stat_statements_available is None:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                    """)
                    self._pg_stat_statements_available = cur.fetchone() is not None
            except Exception:
                self._pg_stat_statements_available = False
        return self._pg_stat_statements_available

    def _get_database_stats(self) -> dict:
        """Get database-level statistics."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    blks_read,
                    blks_hit,
                    xact_commit,
                    xact_rollback,
                    tup_returned,
                    tup_fetched,
                    tup_inserted,
                    tup_updated,
                    tup_deleted,
                    temp_files,
                    temp_bytes,
                    deadlocks,
                    blk_read_time,
                    blk_write_time
                FROM pg_stat_database
                WHERE datname = current_database()
            """)
            row = cur.fetchone()
            if row:
                blks_read, blks_hit = row[0], row[1]
                hit_ratio = blks_hit / (blks_hit + blks_read) if (blks_hit + blks_read) > 0 else 0
                return {
                    "blks_read": blks_read,
                    "blks_hit": blks_hit,
                    "cache_hit_ratio": round(hit_ratio, 4),
                    "xact_commit": row[2],
                    "xact_rollback": row[3],
                    "tup_returned": row[4],
                    "tup_fetched": row[5],
                    "tup_inserted": row[6],
                    "tup_updated": row[7],
                    "tup_deleted": row[8],
                    "temp_files": row[9],
                    "temp_bytes": row[10],
                    "deadlocks": row[11],
                    "blk_read_time_ms": row[12],
                    "blk_write_time_ms": row[13],
                }
        return {}

    def _get_pg_version(self) -> int:
        """Get PostgreSQL major version number."""
        if not hasattr(self, '_pg_version'):
            with self.conn.cursor() as cur:
                cur.execute("SHOW server_version_num")
                result = cur.fetchone()
                # server_version_num is like 170000 for PG17, 160000 for PG16
                self._pg_version = int(result[0]) // 10000 if result else 0
        return self._pg_version

    def _get_bgwriter_stats(self) -> dict:
        """Get background writer and checkpoint statistics."""
        pg_version = self._get_pg_version()

        result = {}

        # Checkpoint stats moved to pg_stat_checkpointer in PG17
        if pg_version >= 17:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        num_timed,
                        num_requested,
                        write_time,
                        sync_time,
                        buffers_written
                    FROM pg_stat_checkpointer
                """)
                row = cur.fetchone()
                if row:
                    result.update({
                        "checkpoints_timed": row[0],
                        "checkpoints_req": row[1],
                        "checkpoint_write_time_ms": row[2],
                        "checkpoint_sync_time_ms": row[3],
                        "buffers_checkpoint": row[4],
                    })

            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        buffers_clean,
                        buffers_alloc
                    FROM pg_stat_bgwriter
                """)
                row = cur.fetchone()
                if row:
                    result.update({
                        "buffers_clean": row[0],
                        "buffers_alloc": row[1],
                    })
        else:
            # PG16 and earlier - all stats in pg_stat_bgwriter
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        checkpoints_timed,
                        checkpoints_req,
                        checkpoint_write_time,
                        checkpoint_sync_time,
                        buffers_checkpoint,
                        buffers_clean,
                        buffers_backend,
                        buffers_backend_fsync,
                        buffers_alloc
                    FROM pg_stat_bgwriter
                """)
                row = cur.fetchone()
                if row:
                    result = {
                        "checkpoints_timed": row[0],
                        "checkpoints_req": row[1],
                        "checkpoint_write_time_ms": row[2],
                        "checkpoint_sync_time_ms": row[3],
                        "buffers_checkpoint": row[4],
                        "buffers_clean": row[5],
                        "buffers_backend": row[6],
                        "buffers_backend_fsync": row[7],
                        "buffers_alloc": row[8],
                    }

        return result

    def _get_index_stats(self, table_name: str) -> dict:
        """Get statistics for the vector index on the specified table."""
        with self.conn.cursor() as cur:
            # Get index usage stats
            cur.execute("""
                SELECT
                    indexrelname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_relation_size(indexrelid) as index_size_bytes
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
                  AND (indexrelname LIKE %s OR indexrelname LIKE %s)
            """, (f"{table_name}%", f"%embedding%",))

            indexes = {}
            for row in cur.fetchall():
                idx_name = row[0]
                indexes[idx_name] = {
                    "idx_scan": row[1],
                    "idx_tup_read": row[2],
                    "idx_tup_fetch": row[3],
                    "index_size_bytes": row[4],
                }

            # Get index IO stats
            cur.execute("""
                SELECT
                    indexrelname,
                    idx_blks_read,
                    idx_blks_hit
                FROM pg_statio_user_indexes
                WHERE schemaname = 'public'
                  AND (indexrelname LIKE %s OR indexrelname LIKE %s)
            """, (f"{table_name}%", f"%embedding%",))

            for row in cur.fetchall():
                idx_name = row[0]
                if idx_name in indexes:
                    blks_read, blks_hit = row[1], row[2]
                    hit_ratio = blks_hit / (blks_hit + blks_read) if (blks_hit + blks_read) > 0 else 0
                    indexes[idx_name].update({
                        "idx_blks_read": blks_read,
                        "idx_blks_hit": blks_hit,
                        "idx_cache_hit_ratio": round(hit_ratio, 4),
                    })

            return indexes

    def _get_table_stats(self, table_name: str) -> dict:
        """Get statistics for the specified table."""
        with self.conn.cursor() as cur:
            # Check if table exists first
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (table_name,))
            if not cur.fetchone()[0]:
                return {}

            cur.execute("""
                SELECT
                    COALESCE(seq_scan, 0),
                    COALESCE(seq_tup_read, 0),
                    COALESCE(idx_scan, 0),
                    COALESCE(idx_tup_fetch, 0),
                    COALESCE(n_tup_ins, 0),
                    COALESCE(n_tup_upd, 0),
                    COALESCE(n_tup_del, 0),
                    COALESCE(n_live_tup, 0),
                    COALESCE(n_dead_tup, 0),
                    pg_relation_size(%s) as table_size_bytes,
                    pg_total_relation_size(%s) as total_size_bytes
                FROM pg_stat_user_tables
                WHERE schemaname = 'public' AND relname = %s
            """, (table_name, table_name, table_name))

            row = cur.fetchone()
            if row:
                return {
                    "seq_scan": row[0],
                    "seq_tup_read": row[1],
                    "idx_scan": row[2],
                    "idx_tup_fetch": row[3],
                    "n_tup_ins": row[4],
                    "n_tup_upd": row[5],
                    "n_tup_del": row[6],
                    "n_live_tup": row[7],
                    "n_dead_tup": row[8],
                    "table_size_bytes": row[9],
                    "total_size_bytes": row[10],
                }
        return {}

    def _get_table_io_stats(self, table_name: str) -> dict:
        """Get IO statistics for the specified table."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    heap_blks_read,
                    heap_blks_hit,
                    idx_blks_read,
                    idx_blks_hit,
                    toast_blks_read,
                    toast_blks_hit
                FROM pg_statio_user_tables
                WHERE schemaname = 'public' AND relname = %s
            """, (table_name,))

            row = cur.fetchone()
            if row:
                heap_read, heap_hit = row[0] or 0, row[1] or 0
                heap_ratio = heap_hit / (heap_hit + heap_read) if (heap_hit + heap_read) > 0 else 0
                return {
                    "heap_blks_read": heap_read,
                    "heap_blks_hit": heap_hit,
                    "heap_cache_hit_ratio": round(heap_ratio, 4),
                    "table_idx_blks_read": row[2] or 0,
                    "table_idx_blks_hit": row[3] or 0,
                    "toast_blks_read": row[4] or 0,
                    "toast_blks_hit": row[5] or 0,
                }
        return {}

    def _get_active_connections(self) -> dict:
        """Get connection and activity statistics."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    count(*) FILTER (WHERE state = 'active') as active,
                    count(*) FILTER (WHERE state = 'idle') as idle,
                    count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
                    count(*) FILTER (WHERE wait_event_type IS NOT NULL) as waiting,
                    count(*) as total
                FROM pg_stat_activity
                WHERE datname = current_database()
            """)
            row = cur.fetchone()
            if row:
                return {
                    "connections_active": row[0],
                    "connections_idle": row[1],
                    "connections_idle_in_transaction": row[2],
                    "connections_waiting": row[3],
                    "connections_total": row[4],
                }
        return {}

    def _get_wait_events(self) -> dict:
        """Get current wait event distribution."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    wait_event_type,
                    wait_event,
                    count(*)
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND state = 'active'
                  AND wait_event IS NOT NULL
                GROUP BY wait_event_type, wait_event
                ORDER BY count(*) DESC
                LIMIT 10
            """)

            wait_events = []
            for row in cur.fetchall():
                wait_events.append({
                    "type": row[0],
                    "event": row[1],
                    "count": row[2],
                })
            return {"top_wait_events": wait_events}

    def capture_snapshot(self, phase: str, table_name: Optional[str] = None) -> dict:
        """
        Capture a complete stats snapshot at a phase boundary.

        Args:
            phase: Name of the phase (e.g., 'baseline', 'after_load', 'after_index')
            table_name: Optional table name for table/index specific stats

        Returns:
            Dictionary containing all captured statistics
        """
        snapshot = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "database": self._get_database_stats(),
            "bgwriter": self._get_bgwriter_stats(),
            "connections": self._get_active_connections(),
            "wait_events": self._get_wait_events(),
        }

        if table_name:
            snapshot["table"] = self._get_table_stats(table_name)
            snapshot["table_io"] = self._get_table_io_stats(table_name)
            snapshot["indexes"] = self._get_index_stats(table_name)

        self.snapshots[phase] = snapshot
        return snapshot

    def compute_delta(self, phase_before: str, phase_after: str) -> dict:
        """
        Compute the difference between two snapshots.

        Args:
            phase_before: Name of the earlier phase
            phase_after: Name of the later phase

        Returns:
            Dictionary containing deltas for numeric metrics
        """
        if phase_before not in self.snapshots or phase_after not in self.snapshots:
            return {}

        before = self.snapshots[phase_before]
        after = self.snapshots[phase_after]

        delta = {
            "phases": f"{phase_before} -> {phase_after}",
            "database": {},
            "bgwriter": {},
            "table": {},
        }

        # Compute database deltas
        for key in ["blks_read", "blks_hit", "xact_commit", "tup_inserted",
                    "temp_files", "temp_bytes", "deadlocks"]:
            if key in before.get("database", {}) and key in after.get("database", {}):
                before_val = before["database"][key]
                after_val = after["database"][key]
                if before_val is not None and after_val is not None:
                    delta["database"][key] = after_val - before_val

        # Cache hit ratio is a point-in-time metric, show both
        if "cache_hit_ratio" in after.get("database", {}):
            delta["database"]["cache_hit_ratio_after"] = after["database"]["cache_hit_ratio"]

        # Compute bgwriter deltas
        for key in ["checkpoints_timed", "checkpoints_req", "buffers_checkpoint",
                    "buffers_clean", "buffers_backend", "checkpoint_write_time_ms"]:
            if key in before.get("bgwriter", {}) and key in after.get("bgwriter", {}):
                before_val = before["bgwriter"][key]
                after_val = after["bgwriter"][key]
                if before_val is not None and after_val is not None:
                    delta["bgwriter"][key] = after_val - before_val

        # Compute table deltas if available
        if before.get("table") and after.get("table"):
            for key in ["seq_scan", "idx_scan", "n_tup_ins"]:
                if key in before["table"] and key in after["table"]:
                    before_val = before["table"][key]
                    after_val = after["table"][key]
                    if before_val is not None and after_val is not None:
                        delta["table"][key] = after_val - before_val

        return delta

    def get_summary(self) -> dict:
        """
        Get a summary of all captured snapshots and key deltas.

        Returns:
            Dictionary with summary statistics
        """
        if not self.snapshots:
            return {}

        phases = list(self.snapshots.keys())
        summary = {
            "phases_captured": phases,
            "snapshots": self.snapshots,
            "deltas": {},
        }

        # Compute deltas between consecutive phases
        for i in range(len(phases) - 1):
            delta_name = f"{phases[i]}_to_{phases[i+1]}"
            summary["deltas"][delta_name] = self.compute_delta(phases[i], phases[i+1])

        # Compute overall delta (first to last)
        if len(phases) >= 2:
            summary["deltas"]["overall"] = self.compute_delta(phases[0], phases[-1])

        return summary

    def format_for_report(self) -> str:
        """
        Format the collected stats as markdown for the report.

        Returns:
            Markdown-formatted string
        """
        lines = []

        # Include custom settings first
        custom_settings_md = self.format_custom_settings()
        if custom_settings_md:
            lines.append(custom_settings_md)

        summary = self.get_summary()
        if not summary:
            return "\n".join(lines) if lines else ""

        lines.extend([
            "## PostgreSQL Statistics",
            "",
        ])

        # Show final snapshot stats
        if summary["snapshots"]:
            last_phase = list(summary["snapshots"].keys())[-1]
            last_snapshot = summary["snapshots"][last_phase]

            final_rows = []
            db = last_snapshot.get("database", {})
            if db:
                final_rows.extend([
                    ["Cache Hit Ratio", f"{db.get('cache_hit_ratio', 0):.2%}"],
                    ["Blocks Read", f"{db.get('blks_read', 0):,}"],
                    ["Blocks Hit", f"{db.get('blks_hit', 0):,}"],
                    ["Temp Files", f"{db.get('temp_files', 0):,}"],
                    ["Deadlocks", f"{db.get('deadlocks', 0):,}"],
                ])

            bgw = last_snapshot.get("bgwriter", {})
            if bgw:
                final_rows.extend([
                    ["Checkpoints (timed)", f"{bgw.get('checkpoints_timed', 0):,}"],
                    ["Checkpoints (requested)", f"{bgw.get('checkpoints_req', 0):,}"],
                    ["Buffers Written (checkpoint)", f"{bgw.get('buffers_checkpoint', 0):,}"],
                ])

            if final_rows:
                lines.extend([
                    f"### Final State ({last_phase})",
                    "",
                ])
                lines.extend(format_markdown_table(["Metric", "Value"], final_rows))

        # Show per-phase deltas
        deltas = summary.get("deltas", {})

        # Define phase display names
        phase_names = {
            "baseline_to_after_load": "Data Loading",
            "after_load_to_after_index": "Index Building",
            "after_index_to_after_benchmark": "Query Benchmark",
            "baseline_to_after_index": "Data Loading + Index Building",
            "baseline_to_after_benchmark": "Total",
        }

        # Show phase-by-phase breakdown
        phase_deltas = [(k, v) for k, v in deltas.items() if k != "overall"]
        if phase_deltas:
            phase_rows = []
            for delta_key, delta in phase_deltas:
                phase_name = phase_names.get(delta_key, delta_key)
                db = delta.get("database", {})
                bgw = delta.get("bgwriter", {})
                checkpoints = bgw.get("checkpoints_req", 0) + bgw.get("checkpoints_timed", 0)
                phase_rows.append([
                    phase_name,
                    f"{db.get('blks_read', 0):,}",
                    f"{db.get('blks_hit', 0):,}",
                    f"{db.get('xact_commit', 0):,}",
                    f"{checkpoints:,}",
                ])

            lines.extend([
                "",
                "### Activity by Phase",
                "",
            ])
            lines.extend(format_markdown_table(
                ["Phase", "Blocks Read", "Blocks Hit", "Transactions", "Checkpoints"],
                phase_rows
            ))

        # Show overall totals
        if "overall" in deltas:
            overall = deltas["overall"]
            db_delta = overall.get("database", {})
            bgw_delta = overall.get("bgwriter", {})

            total_rows = []
            if db_delta:
                total_rows.extend([
                    ["Blocks Read", f"{db_delta.get('blks_read', 0):,}"],
                    ["Blocks Hit", f"{db_delta.get('blks_hit', 0):,}"],
                    ["Transactions", f"{db_delta.get('xact_commit', 0):,}"],
                    ["Rows Inserted", f"{db_delta.get('tup_inserted', 0):,}"],
                ])

            if bgw_delta:
                total_checkpoints = bgw_delta.get('checkpoints_req', 0) + bgw_delta.get('checkpoints_timed', 0)
                total_rows.append(["Checkpoints", f"{total_checkpoints:,}"])
                if bgw_delta.get('checkpoint_write_time_ms', 0) > 0:
                    total_rows.append(["Checkpoint Write Time", f"{bgw_delta.get('checkpoint_write_time_ms', 0):,.0f} ms"])

            if total_rows:
                lines.extend([
                    "",
                    "### Total Changes",
                    "",
                ])
                lines.extend(format_markdown_table(["Metric", "Value"], total_rows))

        lines.append("")
        return "\n".join(lines)
