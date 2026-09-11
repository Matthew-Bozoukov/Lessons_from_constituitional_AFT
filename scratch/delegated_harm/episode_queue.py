# ABOUTME: Durable episode ownership for live addition, pause, and drain of inference workers.
# ABOUTME: Claims are atomic and are never automatically stolen after a heartbeat timeout.
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class EpisodeQueue:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.db() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    arm TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending', owner TEXT,
                    claimed_at REAL, finished_at REAL, result_path TEXT, result_hash TEXT,
                    PRIMARY KEY(arm,id));
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY, arm TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'active',
                    heartbeat REAL NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO settings VALUES ('paused','false');
            """)

    @contextmanager
    def db(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def enqueue(self, arm, cells, owners=None):
        owners = owners or {}
        with self.db() as con:
            for cell in cells:
                owner = owners.get(cell['id'])
                con.execute("INSERT INTO jobs(arm,id,payload,state,owner,claimed_at) VALUES(?,?,?,?,?,?)",
                            (arm, cell['id'], json.dumps(cell), 'claimed' if owner else 'pending',
                             owner, time.time() if owner else None))

    def register(self, worker, arm, metadata):
        with self.db() as con:
            con.execute("INSERT INTO workers VALUES(?,?, 'active',?,?)",
                        (worker, arm, time.time(), json.dumps(metadata)))

    def heartbeat(self, worker):
        with self.db() as con:
            con.execute("UPDATE workers SET heartbeat=? WHERE id=?", (time.time(), worker))

    def mode(self, worker, mode):
        assert mode in ('active', 'drain', 'stopped')
        with self.db() as con:
            assert con.execute("UPDATE workers SET mode=? WHERE id=?", (mode, worker)).rowcount == 1

    def pause(self, value):
        with self.db() as con:
            con.execute("UPDATE settings SET value=? WHERE key='paused'", (json.dumps(bool(value)),))

    def claim(self, worker):
        with self.db() as con:
            con.execute("BEGIN IMMEDIATE")
            w = con.execute("SELECT * FROM workers WHERE id=?", (worker,)).fetchone()
            assert w is not None
            if w['mode'] != 'active':
                return 'drain', None
            if json.loads(con.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0]):
                return 'paused', None
            job = con.execute("SELECT * FROM jobs WHERE arm=? AND state='pending' ORDER BY id LIMIT 1",
                              (w['arm'],)).fetchone()
            if job is None:
                return 'empty', None
            changed = con.execute("UPDATE jobs SET state='claimed',owner=?,claimed_at=? WHERE arm=? AND id=? AND state='pending'",
                                  (worker, time.time(), w['arm'], job['id'])).rowcount
            assert changed == 1
            return 'claimed', json.loads(job['payload'])

    def finish(self, worker, arm, item_id, path, sha):
        with self.db() as con:
            changed = con.execute("UPDATE jobs SET state='done',finished_at=?,result_path=?,result_hash=? WHERE arm=? AND id=? AND owner=? AND state='claimed'",
                                  (time.time(), str(Path(path).resolve()), sha, arm, item_id, worker)).rowcount
            assert changed == 1, 'Only the recorded owner may finish an outstanding claim'

    def snapshot(self):
        with self.db() as con:
            return {"jobs": [dict(r) for r in con.execute("SELECT * FROM jobs ORDER BY arm,id")],
                    "workers": [dict(r) for r in con.execute("SELECT * FROM workers ORDER BY id")],
                    "paused": json.loads(con.execute("SELECT value FROM settings WHERE key='paused'").fetchone()[0])}
