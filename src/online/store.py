from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import ONLINE_DB


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS predictions (
  forecast_origin_utc TEXT NOT NULL,
  delivery_start_utc TEXT NOT NULL,
  step INTEGER NOT NULL,
  p10 REAL, p50 REAL, p90 REAL,
  p_up REAL, p_down REAL, p_none REAL,
  p_spike_high REAL, p_spike_low REAL,
  p50_lgb REAL, p50_cat REAL,
  p50_direct REAL, p50_spread REAL,
  reliability REAL,
  model_version TEXT,
  top_drivers TEXT,
  feature_json TEXT NOT NULL,
  context_json TEXT,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (forecast_origin_utc, delivery_start_utc, step)
);
CREATE TABLE IF NOT EXISTS actuals (
  delivery_start_utc TEXT PRIMARY KEY,
  imbalance_price REAL,
  direction INTEGER,
  spot_price REAL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
  forecast_origin_utc TEXT NOT NULL,
  delivery_start_utc TEXT NOT NULL,
  step INTEGER NOT NULL,
  abs_error REAL,
  covered INTEGER,
  direction_brier REAL,
  spike_high_brier REAL,
  spike_low_brier REAL,
  model_version TEXT,
  evaluated_at_utc TEXT NOT NULL,
  PRIMARY KEY (forecast_origin_utc, delivery_start_utc, step)
);
CREATE TABLE IF NOT EXISTS model_events (
  event_time_utc TEXT NOT NULL,
  event_type TEXT NOT NULL,
  old_version TEXT,
  new_version TEXT,
  reason TEXT,
  metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS ingestion_log (
  event_time_utc TEXT NOT NULL,
  source TEXT NOT NULL,
  status TEXT NOT NULL,
  details_json TEXT
);
"""


def _z(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class OnlineStore:
    def __init__(self, path: Path = ONLINE_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)
            # Forward-compatible migration for V4-V7 databases already on disk.
            cols = {r[1] for r in con.execute("PRAGMA table_info(predictions)").fetchall()}
            if "p50_direct" not in cols:
                con.execute("ALTER TABLE predictions ADD COLUMN p50_direct REAL")
            if "p50_spread" not in cols:
                con.execute("ALTER TABLE predictions ADD COLUMN p50_spread REAL")

    @contextmanager
    def connect(self):
        """Open a SQLite connection and always close it on context exit.

        sqlite3.Connection.__exit__ commits/rolls back but does *not* close the
        underlying file handle.  On Windows that can keep the database locked
        and break temporary-database cleanup (and can accumulate handles in a
        long-running daemon).  This wrapper preserves transaction semantics and
        closes deterministically.
        """
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def log_ingestion(self, source: str, status: str, details: dict) -> None:
        now = _z(pd.Timestamp.now(tz="UTC"))
        with self.connect() as con:
            con.execute(
                "INSERT INTO ingestion_log VALUES (?,?,?,?)",
                (now, source, status, json.dumps(details, default=str)),
            )

    def save_predictions(
        self,
        features: pd.DataFrame,
        predictions: pd.DataFrame,
        drivers: pd.Series,
        model_version: str,
        context: dict | None = None,
    ) -> None:
        now = _z(pd.Timestamp.now(tz="UTC"))
        context_json = json.dumps(context or {}, default=str)
        rows = []
        for i in range(len(features)):
            f = features.iloc[i]
            p = predictions.iloc[i]
            feature_payload = {
                k: (None if pd.isna(v) else float(v) if isinstance(v, (bool, np.bool_, int, float, np.number)) else str(v))
                for k, v in f.items()
                if k not in {"forecast_origin_utc", "delivery_start_utc"}
            }
            rows.append((
                _z(f["forecast_origin_utc"]), _z(f["delivery_start_utc"]), int(f["step"]),
                float(p["p10"]), float(p["p50"]), float(p["p90"]),
                float(p["p_up"]), float(p["p_down"]), float(p["p_none"]),
                float(p["p_spike_high"]), float(p["p_spike_low"]),
                float(p.get("p50_lgb", np.nan)), float(p.get("p50_cat", np.nan)),
                float(p.get("p50_direct", p.get("p50", np.nan))),
                float(p.get("p50_spread", np.nan)),
                float(p.get("reliability", np.nan)), model_version, str(drivers.iloc[i]),
                json.dumps(feature_payload, separators=(",", ":")), context_json, now,
            ))
        sql = """
        INSERT OR REPLACE INTO predictions
        (forecast_origin_utc,delivery_start_utc,step,p10,p50,p90,p_up,p_down,p_none,
         p_spike_high,p_spike_low,p50_lgb,p50_cat,p50_direct,p50_spread,reliability,model_version,top_drivers,
         feature_json,context_json,created_at_utc)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self.connect() as con:
            con.executemany(sql, rows)

    def upsert_actuals_from_long(self, live_long: pd.DataFrame) -> None:
        if live_long.empty:
            return
        now = _z(pd.Timestamp.now(tz="UTC"))
        frames = {}
        for series in ["fi_imbalance_price", "fi_mfrr_direction", "fi_spot_price"]:
            g = live_long[live_long["series_id"] == series]
            if not g.empty:
                frames[series] = g.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last").set_index("timestamp_utc")["value"]
        keys = sorted(set().union(*[set(s.index) for s in frames.values()])) if frames else []
        rows = []
        for ts in keys:
            price = frames.get("fi_imbalance_price", pd.Series(dtype=float)).get(ts, np.nan)
            direction = frames.get("fi_mfrr_direction", pd.Series(dtype=float)).get(ts, np.nan)
            spot = frames.get("fi_spot_price", pd.Series(dtype=float)).get(ts, np.nan)
            rows.append((_z(ts), None if pd.isna(price) else float(price), None if pd.isna(direction) else int(direction), None if pd.isna(spot) else float(spot), now))
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO actuals(delivery_start_utc,imbalance_price,direction,spot_price,updated_at_utc)
                VALUES (?,?,?,?,?)
                ON CONFLICT(delivery_start_utc) DO UPDATE SET
                  imbalance_price=COALESCE(excluded.imbalance_price, actuals.imbalance_price),
                  direction=COALESCE(excluded.direction, actuals.direction),
                  spot_price=COALESCE(excluded.spot_price, actuals.spot_price),
                  updated_at_utc=excluded.updated_at_utc
                """,
                rows,
            )

    def evaluate_settled(self, cutoff: pd.Timestamp) -> int:
        cutoff_z = _z(cutoff)
        sql = """
        SELECT p.*, a.imbalance_price, a.direction, a.spot_price
        FROM predictions p
        JOIN actuals a ON a.delivery_start_utc=p.delivery_start_utc
        LEFT JOIN evaluations e
          ON e.forecast_origin_utc=p.forecast_origin_utc
         AND e.delivery_start_utc=p.delivery_start_utc
         AND e.step=p.step
        WHERE p.delivery_start_utc <= ? AND e.forecast_origin_utc IS NULL
          AND a.imbalance_price IS NOT NULL
        """
        with self.connect() as con:
            rows = con.execute(sql, (cutoff_z,)).fetchall()
        if not rows:
            return 0
        now = _z(pd.Timestamp.now(tz="UTC"))
        inserts = []
        for r in rows:
            y = float(r["imbalance_price"])
            direction = r["direction"]
            dir_brier = None
            if direction is not None:
                cls = {-1: 0, 0: 1, 1: 2}.get(int(direction))
                if cls is not None:
                    probs = np.array([r["p_down"], r["p_none"], r["p_up"]], float)
                    obs = np.zeros(3); obs[cls] = 1
                    dir_brier = float(np.sum((probs - obs) ** 2))
            hi_brier = lo_brier = None
            if r["spot_price"] is not None:
                spot = float(r["spot_price"])
                hi = float(y > spot + 50.0)
                lo = float(y < spot - 50.0)
                hi_brier = float((float(r["p_spike_high"]) - hi) ** 2)
                lo_brier = float((float(r["p_spike_low"]) - lo) ** 2)
            inserts.append((
                r["forecast_origin_utc"], r["delivery_start_utc"], r["step"],
                abs(y - float(r["p50"])), int(float(r["p10"]) <= y <= float(r["p90"])),
                dir_brier, hi_brier, lo_brier, r["model_version"], now,
            ))
        with self.connect() as con:
            con.executemany("INSERT OR REPLACE INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?)", inserts)
        return len(inserts)

    def recent_joined(self, since: pd.Timestamp) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql_query(
                """
                SELECT p.*, a.imbalance_price, a.direction, a.spot_price,
                       e.abs_error,e.covered,e.direction_brier,e.spike_high_brier,e.spike_low_brier
                FROM predictions p
                JOIN actuals a ON a.delivery_start_utc=p.delivery_start_utc
                JOIN evaluations e
                  ON e.forecast_origin_utc=p.forecast_origin_utc
                 AND e.delivery_start_utc=p.delivery_start_utc AND e.step=p.step
                WHERE p.delivery_start_utc >= ?
                ORDER BY p.forecast_origin_utc,p.step
                """,
                con,
                params=(_z(since),),
            )
        return df

    def settled_feature_frame(self) -> pd.DataFrame:
        with self.connect() as con:
            df = pd.read_sql_query(
                """
                SELECT p.forecast_origin_utc,p.delivery_start_utc,p.step,p.feature_json,
                       a.imbalance_price,a.direction,a.spot_price
                FROM predictions p JOIN actuals a ON a.delivery_start_utc=p.delivery_start_utc
                WHERE a.imbalance_price IS NOT NULL AND a.direction IS NOT NULL
                ORDER BY p.forecast_origin_utc,p.step
                """, con,
            )
        if df.empty:
            return df
        records = []
        for _, row in df.iterrows():
            feat = json.loads(row["feature_json"])
            feat["forecast_origin_utc"] = pd.Timestamp(row["forecast_origin_utc"])
            feat["delivery_start_utc"] = pd.Timestamp(row["delivery_start_utc"])
            feat["step"] = int(row["step"])
            feat["y_price"] = float(row["imbalance_price"])
            d = int(row["direction"])
            feat["y_direction"] = {-1: 0, 0: 1, 1: 2}[d]
            if pd.notna(row["spot_price"]):
                spot = float(row["spot_price"])
                feat["y_spike_high"] = float(feat["y_price"] > spot + 50.0)
                feat["y_spike_low"] = float(feat["y_price"] < spot - 50.0)
            else:
                feat["y_spike_high"] = np.nan
                feat["y_spike_low"] = np.nan
            records.append(feat)
        out = pd.DataFrame(records)
        out["forecast_origin_utc"] = pd.to_datetime(out["forecast_origin_utc"], utc=True)
        out["delivery_start_utc"] = pd.to_datetime(out["delivery_start_utc"], utc=True)

        # JSON snapshots can reconstruct sparse numeric columns as ``object``.
        # Normalize prediction-time feature/label columns here so comparison and
        # challenger training receive the same numerical contract as live models.
        protected = {"forecast_origin_utc", "delivery_start_utc"}
        bool_map = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0,
                    "true": 1.0, "false": 0.0, "TRUE": 1.0, "FALSE": 0.0}
        for c in out.columns:
            if c in protected:
                continue
            series = out[c]
            if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
                series = series.replace(bool_map)
            out[c] = pd.to_numeric(series, errors="coerce")
        return out

    def log_model_event(self, event_type: str, old_version: str | None, new_version: str | None, reason: str, metrics: dict | None = None) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO model_events VALUES (?,?,?,?,?,?)",
                (_z(pd.Timestamp.now(tz="UTC")), event_type, old_version, new_version, reason, json.dumps(metrics or {}, default=str)),
            )

    def last_model_event(self, event_type: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM model_events WHERE event_type=? ORDER BY event_time_utc DESC LIMIT 1",
                (event_type,),
            ).fetchone()
        return dict(row) if row else None

    def status(self) -> dict:
        with self.connect() as con:
            p = con.execute("SELECT COUNT(*) n, MAX(forecast_origin_utc) latest FROM predictions").fetchone()
            e = con.execute("SELECT COUNT(*) n, AVG(abs_error) mae, AVG(covered) coverage FROM evaluations").fetchone()
            m = con.execute("SELECT * FROM model_events ORDER BY event_time_utc DESC LIMIT 1").fetchone()
        return {
            "predictions": int(p["n"]), "latest_origin": p["latest"],
            "evaluations": int(e["n"]), "mean_abs_error": e["mae"], "coverage": e["coverage"],
            "last_model_event": dict(m) if m else None,
        }
