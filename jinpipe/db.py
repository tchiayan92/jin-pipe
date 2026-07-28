"""SQLite-backed job/state store for the pipeline.

Single-writer discipline: this class is only ever called from one thread - the
orchestrator's asyncio event loop thread. Worker processes (ProcessPoolExecutor
stages and persistent ASR/diarization workers) never touch SQLite directly; they
return plain data over queues/futures, and the main loop is the only thing that
calls into JobStore to persist results. This avoids cross-process lock
contention entirely instead of fighting it with retries.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATUSES = ("PENDING", "RUNNING", "DONE", "FAILED")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    channel TEXT,
    url TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','DONE','FAILED')),
    raw_path TEXT,
    standardized_path TEXT,
    duration_s REAL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS superchunks (
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    idx INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','DONE','FAILED')),
    words_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (video_id, idx)
);

CREATE TABLE IF NOT EXISTS segments (
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    idx INTEGER NOT NULL,
    segment_id TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','RUNNING','DONE','FAILED')),
    text TEXT,
    words_json TEXT,
    speaker TEXT,
    exceeds_max_duration INTEGER NOT NULL DEFAULT 0,
    dnsmos_ovr REAL,
    output_audio_path TEXT,
    output_json_path TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (video_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_superchunks_status ON superchunks(video_id, status);
CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=True)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "JobStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- videos ----

    def add_video(self, video_id: str, url: str, channel: str | None = None) -> bool:
        """Insert a video as PENDING if it doesn't already exist. Returns True if newly inserted."""
        now = _now()
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO videos (video_id, channel, url, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'PENDING', ?, ?)",
            (video_id, channel, url, now, now),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_video(self, video_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return dict(row) if row else None

    def list_videos(self, status: str | None = None) -> list[dict]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM videos").fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM videos WHERE status = ?", (status,)).fetchall()
        return [dict(r) for r in rows]

    def update_video(self, video_id: str, **fields) -> None:
        self._update("videos", ("video_id",), (video_id,), fields)

    # ---- superchunks ----

    def add_superchunk(self, video_id: str, idx: int, start_s: float, end_s: float) -> bool:
        now = _now()
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO superchunks (video_id, idx, start_s, end_s, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'PENDING', ?, ?)",
            (video_id, idx, start_s, end_s, now, now),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_superchunks(self, video_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM superchunks WHERE video_id = ? ORDER BY idx", (video_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def update_superchunk(self, video_id: str, idx: int, **fields) -> None:
        self._update("superchunks", ("video_id", "idx"), (video_id, idx), fields)

    def superchunks_all_done(self, video_id: str) -> bool:
        rows = self.get_superchunks(video_id)
        if not rows:
            return False
        return all(r["status"] == "DONE" for r in rows)

    def list_superchunks(self, status: str | None = None) -> list[dict]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM superchunks").fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM superchunks WHERE status = ?", (status,)).fetchall()
        return [dict(r) for r in rows]

    # ---- segments ----

    def add_segment(
        self, video_id: str, idx: int, segment_id: str, start_s: float, end_s: float, **fields
    ) -> bool:
        now = _now()
        base = {
            "text": None,
            "words_json": None,
            "speaker": None,
            "exceeds_max_duration": 0,
            "dnsmos_ovr": None,
            "output_audio_path": None,
            "output_json_path": None,
            "error": None,
        }
        base.update(fields)
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO segments "
            "(video_id, idx, segment_id, start_s, end_s, status, text, words_json, speaker, "
            " exceeds_max_duration, dnsmos_ovr, output_audio_path, output_json_path, error, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                video_id,
                idx,
                segment_id,
                start_s,
                end_s,
                base["text"],
                base["words_json"],
                base["speaker"],
                int(base["exceeds_max_duration"]),
                base["dnsmos_ovr"],
                base["output_audio_path"],
                base["output_json_path"],
                base["error"],
                now,
                now,
            ),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_segment(self, video_id: str, idx: int, **fields) -> None:
        self._update("segments", ("video_id", "idx"), (video_id, idx), fields)

    def get_segments(self, video_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM segments WHERE video_id = ? ORDER BY idx", (video_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_segments(self, status: str | None = None) -> list[dict]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM segments").fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM segments WHERE status = ?", (status,)).fetchall()
        return [dict(r) for r in rows]

    # ---- generic update helper ----

    def _update(self, table: str, key_cols: tuple[str, ...], key_vals: tuple, fields: dict) -> None:
        if not fields:
            return
        fields = dict(fields)
        fields["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        where_clause = " AND ".join(f"{k} = ?" for k in key_cols)
        params = list(fields.values()) + list(key_vals)
        self._conn.execute(f"UPDATE {table} SET {set_clause} WHERE {where_clause}", params)
        self._conn.commit()

    # ---- crash recovery ----

    def recover_stale_running(self, max_age_s: float = 0.0) -> int:
        """Reset RUNNING rows older than max_age_s back to PENDING, across all tables.

        max_age_s=0 treats every currently-RUNNING row as stale, which is correct
        right after a process restart since nothing can genuinely still be running.
        """
        total = 0
        cutoff = None
        if max_age_s > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_s)).isoformat()
        for table in ("videos", "superchunks", "segments"):
            now = _now()
            if cutoff is None:
                cur = self._conn.execute(
                    f"UPDATE {table} SET status = 'PENDING', updated_at = ? WHERE status = 'RUNNING'",
                    (now,),
                )
            else:
                cur = self._conn.execute(
                    f"UPDATE {table} SET status = 'PENDING', updated_at = ? "
                    "WHERE status = 'RUNNING' AND updated_at < ?",
                    (now, cutoff),
                )
            total += cur.rowcount
        self._conn.commit()
        return total

    # ---- status reporting ----

    def status_counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for table in ("videos", "superchunks", "segments"):
            rows = self._conn.execute(f"SELECT status, COUNT(*) AS n FROM {table} GROUP BY status").fetchall()
            result[table] = {r["status"]: r["n"] for r in rows}
        return result

    def print_status(self, console) -> None:
        from rich.table import Table

        counts = self.status_counts()
        table = Table(title="JinPipe job store status")
        table.add_column("stage")
        for s in STATUSES:
            table.add_column(s)
        for stage_table, by_status in counts.items():
            table.add_row(stage_table, *(str(by_status.get(s, 0)) for s in STATUSES))
        console.print(table)
