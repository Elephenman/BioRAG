"""BioRAG SQLite 元数据管理"""

import os
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime


class MetadataDB:
    """SQLite 元数据管理"""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                source      TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                category    TEXT NOT NULL,
                file_type   TEXT NOT NULL,
                folder      TEXT,
                subfolder   TEXT,
                file_size   INTEGER,
                file_mtime  TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                char_count  INTEGER DEFAULT 0,
                indexed_at  TEXT NOT NULL,
                file_hash   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_files_category ON files(category);
            CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type);

            CREATE VIEW IF NOT EXISTS category_stats AS
            SELECT
                category,
                COUNT(*) AS file_count,
                SUM(chunk_count) AS chunk_count,
                SUM(char_count) AS total_chars,
                SUM(file_size) AS total_size
            FROM files
            GROUP BY category;
        """)
        self.conn.commit()

    def upsert_file(self, source: str, filename: str, category: str, file_type: str,
                    folder: str, subfolder: str, file_size: int, file_mtime: str,
                    chunk_count: int, char_count: int, file_hash: str):
        """插入或更新文件记录"""
        self.conn.execute("""
            INSERT OR REPLACE INTO files
            (source, filename, category, file_type, folder, subfolder,
             file_size, file_mtime, chunk_count, char_count, indexed_at, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (source, filename, category, file_type, folder, subfolder,
              file_size, file_mtime, chunk_count, char_count,
              datetime.now().isoformat(), file_hash))
        self.conn.commit()

    def get_file(self, source: str) -> Optional[Dict]:
        """获取文件记录"""
        row = self.conn.execute("SELECT * FROM files WHERE source = ?", (source,)).fetchone()
        return dict(row) if row else None

    def delete_file(self, source: str):
        """删除文件记录"""
        self.conn.execute("DELETE FROM files WHERE source = ?", (source,))
        self.conn.commit()

    def get_all_hashes(self) -> Dict[str, str]:
        """获取所有文件的哈希值 {source: hash}"""
        rows = self.conn.execute("SELECT source, file_hash FROM files").fetchall()
        return {row["source"]: row["file_hash"] for row in rows}

    def get_category_stats(self) -> Dict:
        """获取分类统计"""
        rows = self.conn.execute("SELECT * FROM category_stats").fetchall()
        stats = {}
        for row in rows:
            stats[row["category"]] = {
                "file_count": row["file_count"],
                "chunk_count": row["chunk_count"],
                "total_chars": row["total_chars"] or 0,
                "total_size": row["total_size"] or 0,
            }
        return stats

    def get_total_stats(self) -> Dict:
        """获取总体统计"""
        row = self.conn.execute("""
            SELECT COUNT(*) as doc_count,
                   SUM(chunk_count) as total_chunks,
                   SUM(file_size) as total_size
            FROM files
        """).fetchone()
        return {
            "total_documents": row["doc_count"] or 0,
            "total_chunks": row["total_chunks"] or 0,
            "total_size": row["total_size"] or 0,
        }

    def update_paths(self, old_path: str, new_path: str):
        """迁移时更新所有路径"""
        self.conn.execute(
            "UPDATE files SET source = REPLACE(source, ?, ?) WHERE source LIKE ?",
            (old_path, new_path, old_path + "%")
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
