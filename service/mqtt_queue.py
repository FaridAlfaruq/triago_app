"""SQLite queue shared by the Raspi outbox and server inbox."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time


class MessageQueue:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS messages (
                event_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                digest TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0)""")

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def put(self, envelope):
        payload = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connection() as conn:
            old = conn.execute("SELECT digest FROM messages WHERE event_id = ?",
                               (envelope["event_id"],)).fetchone()
            if old:
                if old["digest"] != digest:
                    raise ValueError("event_id digunakan untuk payload yang berbeda")
                return False
            conn.execute("INSERT INTO messages(event_id, payload, digest) VALUES (?, ?, ?)",
                         (envelope["event_id"], payload, digest))
        return True

    def next_message(self):
        with self.connection() as conn:
            row = conn.execute("""SELECT * FROM messages WHERE done = 0 AND next_attempt <= ?
                ORDER BY rowid LIMIT 1""", (time.time(),)).fetchone()
        return dict(row) if row else None

    def complete(self, event_id):
        # Keep the ID/hash for deduplication; discard completed patient payloads.
        with self.connection() as conn:
            conn.execute("UPDATE messages SET done = 1, payload = '' WHERE event_id = ?", (event_id,))

    def retry(self, event_id):
        with self.connection() as conn:
            row = conn.execute("SELECT attempts FROM messages WHERE event_id = ?", (event_id,)).fetchone()
            delay = min(60, 2 ** min(row["attempts"] + 1, 6))
            conn.execute("UPDATE messages SET attempts = attempts + 1, next_attempt = ? WHERE event_id = ?",
                         (time.time() + delay, event_id))

    def pending_count(self):
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM messages WHERE done = 0").fetchone()[0]
