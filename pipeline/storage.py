"""
Persistence layer: per-meal logs + incrementally-updated daily rollups.

SQLite is enough for a single-user demo / small deployment; swap the
connection string for Postgres in `Storage.__init__` for multi-user
production use -- the schema and queries are plain SQL and portable.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path
from typing import Optional

from .nutrition_db import MacroResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    kcal_target REAL,
    protein_target_g REAL,
    fat_target_g REAL,
    carbs_target_g REAL
);

CREATE TABLE IF NOT EXISTS meals (
    meal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    logged_at TEXT NOT NULL,   -- ISO8601
    date TEXT NOT NULL,        -- YYYY-MM-DD, derived, indexed for daily rollups
    items_json TEXT NOT NULL,  -- list of {name, grams, kcal, protein_g, ...}
    kcal REAL NOT NULL,
    protein_g REAL NOT NULL,
    fat_g REAL NOT NULL,
    carbs_g REAL NOT NULL,
    user_corrected INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals(user_id, date);

CREATE TABLE IF NOT EXISTS daily_summary (
    user_id TEXT NOT NULL,
    date TEXT NOT NULL,
    kcal REAL NOT NULL DEFAULT 0,
    protein_g REAL NOT NULL DEFAULT 0,
    fat_g REAL NOT NULL DEFAULT 0,
    carbs_g REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, date)
);
"""


@dataclasses.dataclass
class UserTargets:
    user_id: str
    kcal_target: float
    protein_target_g: float
    fat_target_g: float
    carbs_target_g: float


class Storage:
    def __init__(self, db_path: str = "nutrivision.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_user(self, targets: UserTargets) -> None:
        self.conn.execute(
            """INSERT INTO users (user_id, kcal_target, protein_target_g, fat_target_g, carbs_target_g)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 kcal_target=excluded.kcal_target,
                 protein_target_g=excluded.protein_target_g,
                 fat_target_g=excluded.fat_target_g,
                 carbs_target_g=excluded.carbs_target_g""",
            (targets.user_id, targets.kcal_target, targets.protein_target_g,
             targets.fat_target_g, targets.carbs_target_g),
        )
        self.conn.commit()

    def log_meal(
        self,
        user_id: str,
        logged_at_iso: str,
        items: list[MacroResult],
        totals: MacroResult,
        user_corrected: bool = False,
    ) -> int:
        date = logged_at_iso[:10]
        items_json = json.dumps(
            [
                {
                    "name": it.food_name,
                    "grams": round(it.grams, 1),
                    "kcal": round(it.kcal, 1),
                    "protein_g": round(it.protein_g, 1),
                    "fat_g": round(it.fat_g, 1),
                    "carbs_g": round(it.carbs_g, 1),
                }
                for it in items
            ]
        )
        cur = self.conn.execute(
            """INSERT INTO meals (user_id, logged_at, date, items_json, kcal, protein_g, fat_g, carbs_g, user_corrected)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, logged_at_iso, date, items_json, totals.kcal, totals.protein_g,
             totals.fat_g, totals.carbs_g, int(user_corrected)),
        )
        meal_id = cur.lastrowid

        # Incremental rollup -- avoids re-summing the whole day's meals on every insert.
        self.conn.execute(
            """INSERT INTO daily_summary (user_id, date, kcal, protein_g, fat_g, carbs_g)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, date) DO UPDATE SET
                 kcal = kcal + excluded.kcal,
                 protein_g = protein_g + excluded.protein_g,
                 fat_g = fat_g + excluded.fat_g,
                 carbs_g = carbs_g + excluded.carbs_g""",
            (user_id, date, totals.kcal, totals.protein_g, totals.fat_g, totals.carbs_g),
        )
        self.conn.commit()
        return meal_id

    def get_daily_summary(self, user_id: str, date: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM daily_summary WHERE user_id=? AND date=?", (user_id, date)
        ).fetchone()
        summary = dict(row) if row else {"user_id": user_id, "date": date, "kcal": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0}

        user_row = self.conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if user_row:
            summary["kcal_target"] = user_row["kcal_target"]
            summary["protein_target_g"] = user_row["protein_target_g"]
            summary["fat_target_g"] = user_row["fat_target_g"]
            summary["carbs_target_g"] = user_row["carbs_target_g"]
            summary["kcal_remaining"] = user_row["kcal_target"] - summary["kcal"]
            summary["protein_remaining_g"] = user_row["protein_target_g"] - summary["protein_g"]
        return summary

    def get_meals(self, user_id: str, date: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM meals WHERE user_id=? AND date=? ORDER BY logged_at", (user_id, date)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["items"] = json.loads(d.pop("items_json"))
            out.append(d)
        return out
