"""
Stores 👍/👎 feedback per (query, image_id) pair in SQLite.

How this improves ranking WITHOUT retraining the CLIP model (see README
for the full explanation):
  1. Short-term / immediate: at rerank time, boost or penalize final_score
     for an image based on its historical net feedback for similar queries
     (see get_feedback_boost below) — a cheap lookup, no training loop.
  2. Medium-term: periodically mine (query_attributes, image_id, label)
     rows to auto-tune RANK_WEIGHTS (e.g. grid search / logistic regression
     over the 3 weight scalars) — still not touching the CLIP model itself.
  3. Long-term (optional, out of scope for this deployment): use feedback
     as weak supervision to fine-tune a small linear probe on top of frozen
     CLIP embeddings — still frozen backbone, cheap to train, cheap to swap.
"""
import sqlite3
import time

from utils.config import FEEDBACK_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    image_id TEXT NOT NULL,
    label INTEGER NOT NULL,      -- 1 = relevant, -1 = not relevant
    created_at REAL NOT NULL
);
"""


def _connect():
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def record_feedback(query_text, image_id, relevant: bool):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO feedback (query_text, image_id, label, created_at) VALUES (?, ?, ?, ?)",
            (query_text, image_id, 1 if relevant else -1, time.time()),
        )
    conn.close()


def get_feedback_boost(image_id, alpha=0.02, cap=0.1):
    """
    Small additive nudge to final_score based on this image's net
    historical feedback across ALL queries. Deliberately small and capped
    so a handful of votes can't override embedding/attribute signal.
    """
    conn = _connect()
    row = conn.execute(
        "SELECT SUM(label) FROM feedback WHERE image_id = ?", (image_id,)
    ).fetchone()
    conn.close()
    net = row[0] or 0
    return max(-cap, min(cap, alpha * net))
