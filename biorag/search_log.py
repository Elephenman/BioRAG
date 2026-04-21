"""BioRAG 检索日志记录"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional


class SearchLogDB:
    """检索日志记录"""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                query         TEXT NOT NULL,
                level         INTEGER DEFAULT 1,
                category      TEXT,
                top_k         INTEGER DEFAULT 5,
                results_count INTEGER,
                avg_score     REAL,
                min_score     REAL,
                max_score     REAL,
                time_ms       INTEGER,
                sources       TEXT,
                created_at    TEXT DEFAULT (datetime('now', 'localtime'))
            );
            CREATE INDEX IF NOT EXISTS idx_logs_date ON search_logs(created_at);
            CREATE INDEX IF NOT EXISTS idx_logs_query ON search_logs(query);
        """)
        self.conn.commit()

    def log_search(self, query: str, level: int, category: Optional[str],
                   top_k: int, results_count: int, avg_score: float,
                   min_score: float, max_score: float, time_ms: int,
                   sources: List[str]):
        """记录一次检索"""
        self.conn.execute("""
            INSERT INTO search_logs
            (query, level, category, top_k, results_count,
             avg_score, min_score, max_score, time_ms, sources)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (query, level, category, top_k, results_count,
              avg_score, min_score, max_score, time_ms, json.dumps(sources)))
        self.conn.commit()

    def get_recent_stats(self, days: int = 7) -> Dict:
        """获取近N天检索统计"""
        row = self.conn.execute("""
            SELECT
                COUNT(*) AS total_searches,
                ROUND(AVG(avg_score), 3) AS avg_similarity,
                SUM(CASE WHEN avg_score < 0.5 THEN 1 ELSE 0 END) AS low_score_count,
                ROUND(AVG(time_ms), 0) AS avg_time_ms
            FROM search_logs
            WHERE created_at >= datetime('now', '-' || ? || ' days', 'localtime')
        """, (days,)).fetchone()

        return {
            "total_searches": row["total_searches"] or 0,
            "avg_similarity": row["avg_similarity"] or 0,
            "low_score_count": row["low_score_count"] or 0,
            "avg_time_ms": row["avg_time_ms"] or 0,
        }

    def get_daily_stats(self, days: int = 7) -> List[Dict]:
        """获取每日检索统计"""
        rows = self.conn.execute("""
            SELECT
                DATE(created_at) AS date,
                COUNT(*) AS count,
                ROUND(AVG(avg_score), 3) AS avg_score,
                ROUND(AVG(time_ms), 0) AS avg_ms
            FROM search_logs
            WHERE created_at >= datetime('now', '-' || ? || ' days', 'localtime')
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """, (days,)).fetchall()
        return [dict(r) for r in rows]

    def get_hot_files(self, days: int = 7, limit: int = 10) -> List[Dict]:
        """获取热门文件"""
        rows = self.conn.execute("""
            SELECT
                source,
                COUNT(*) AS hit_count
            FROM search_logs, json_each(search_logs.sources)
            WHERE created_at >= datetime('now', '-' || ? || ' days', 'localtime')
            GROUP BY source
            ORDER BY hit_count DESC
            LIMIT ?
        """, (days, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_low_score_queries(self, days: int = 7, threshold: float = 0.5) -> List[Dict]:
        """获取低分检索查询"""
        rows = self.conn.execute("""
            SELECT
                query,
                ROUND(AVG(avg_score), 3) AS avg_score,
                COUNT(*) AS count
            FROM search_logs
            WHERE created_at >= datetime('now', '-' || ? || ' days', 'localtime')
              AND avg_score < ?
            GROUP BY query
            ORDER BY avg_score ASC
            LIMIT 20
        """, (days, threshold)).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
