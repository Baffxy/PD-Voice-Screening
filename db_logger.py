"""
Logs every prediction the API makes to a Neon (Postgres) database, so usage
persists across API restarts/redeploys — unlike Render's ephemeral filesystem.

A connection is opened per call rather than pooled: prediction volume here is
low, and Neon's scale-to-zero means an idle connection would go stale between
requests anyway. This keeps things simple and correct for this project's scale.
"""

import os
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")


def _get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — cannot log predictions.")
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Create the predictions table if it doesn't already exist. Safe to call on every startup."""
    if not DATABASE_URL:
        print("Warning: DATABASE_URL not set — prediction logging is disabled.")
        return

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    endpoint TEXT NOT NULL,
                    mdvp_fo_hz REAL,
                    mdvp_jitter_pct REAL,
                    mdvp_shimmer REAL,
                    hnr REAL,
                    ppe REAL,
                    prediction TEXT,
                    probability_parkinsons REAL
                )
            """)
        conn.commit()
    finally:
        conn.close()


def get_stats() -> dict:
    """Basic aggregate stats over all logged predictions — the visible payoff of monitoring."""
    if not DATABASE_URL:
        return {"error": "Prediction logging is not configured (DATABASE_URL not set)."}

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM predictions")
            total = cur.fetchone()[0]

            if total == 0:
                return {"total_predictions": 0}

            cur.execute("SELECT endpoint, COUNT(*) FROM predictions GROUP BY endpoint")
            by_endpoint = dict(cur.fetchall())

            cur.execute("SELECT AVG(probability_parkinsons) FROM predictions")
            avg_probability = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM predictions WHERE prediction = 'Elevated risk indicators detected'"
            )
            elevated_count = cur.fetchone()[0]

            return {
                "total_predictions": total,
                "by_endpoint": by_endpoint,
                "average_probability": round(float(avg_probability), 4) if avg_probability is not None else None,
                "elevated_risk_count": elevated_count,
                "no_elevated_risk_count": total - elevated_count,
            }
    finally:
        conn.close()


def log_prediction(endpoint: str, feature_values: dict, prediction: str, probability: float):
    """Insert one prediction record. Failures are logged but never crash the API —
    logging is a nice-to-have, not something that should break a user's actual request."""
    if not DATABASE_URL:
        return

    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                        (timestamp, endpoint, mdvp_fo_hz, mdvp_jitter_pct, mdvp_shimmer, hnr, ppe, prediction, probability_parkinsons)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.now(timezone.utc),
                        endpoint,
                        feature_values.get("MDVP:Fo(Hz)"),
                        feature_values.get("MDVP:Jitter(%)"),
                        feature_values.get("MDVP:Shimmer"),
                        feature_values.get("HNR"),
                        feature_values.get("PPE"),
                        prediction,
                        probability,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"Warning: failed to log prediction: {e}")