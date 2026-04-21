"""Incremental updater: only process new or changed files."""

import hashlib
import json
import logging
import os
import time

from biorag.config import BioRAGConfig, get_category_for_path
from biorag.loader import scan_knowledge_base, load_file
from biorag.chunker import chunk_document, Chunk
from biorag.embedder import BioRAGEmbedder
from biorag.metadata import MetadataDB
from biorag.vectorstore import BioRAGVectorStore

logger = logging.getLogger(__name__)


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _load_file_index(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": "2.0", "files": {}}


def _save_file_index(path: str, index: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def incremental_update(config: BioRAGConfig, vectorstore: BioRAGVectorStore,
                       metadata_db: MetadataDB, embedder: BioRAGEmbedder,
                       force: bool = False) -> dict:
    """Incrementally update the index."""
    from datetime import datetime

    if force:
        from biorag.builder import build_index
        return build_index(config, force=True)

    start_time = time.time()

    data_dir = os.path.abspath(config.data_dir)
    index_path = os.path.join(data_dir, "file_index.json")

    files = scan_knowledge_base(config.knowledge_base_path, config.knowledge_base_ignore)
    file_index = _load_file_index(index_path)

    new_files = 0
    updated_files = 0
    deleted_files = 0
    new_chunks = 0

    # Check for deleted files
    indexed_sources = set(file_index["files"].keys())
    current_sources = set(files)
    for removed in indexed_sources - current_sources:
        old_info = file_index["files"].get(removed, {})
        vectorstore.delete_by_source(removed)
        metadata_db.delete_file(removed)
        del file_index["files"][removed]
        deleted_files += 1

    # Process new and changed files
    existing_hashes = metadata_db.get_all_hashes()

    for filepath in files:
        fhash = _file_hash(filepath)
        old_hash = existing_hashes.get(filepath)

        if old_hash == fhash and file_index["files"].get(filepath, {}).get("status") == "indexed":
            continue

        try:
            doc = load_file(filepath, config.knowledge_base_path)
            if doc is None:
                continue

            category = get_category_for_path(filepath, config.categories)

            chunks = chunk_document(doc, config.chunking)

            if not chunks:
                continue

            # Fill category in chunk metadata
            for c in chunks:
                if not c.metadata.get("category"):
                    c.metadata["category"] = category

            if old_hash:
                vectorstore.delete_by_source(filepath)
                updated_files += 1
            else:
                new_files += 1

            vectorstore.add_documents(chunks)
            new_chunks += len(chunks)

            now = datetime.now().isoformat()
            file_index["files"][filepath] = {
                "hash": fhash,
                "mtime": doc.file_mtime,
                "chunk_count": len(chunks),
                "indexed_at": now,
                "status": "indexed",
            }

            metadata_db.upsert_file(
                source=doc.source, filename=doc.filename, category=category,
                file_type=doc.file_type, folder=doc.folder, subfolder=doc.subfolder,
                file_size=doc.file_size, file_mtime=doc.file_mtime,
                chunk_count=len(chunks), char_count=sum(c.metadata.get("char_count", 0) for c in chunks),
                file_hash=fhash,
            )

        except Exception as e:
            logger.error(f"Failed to update {filepath}: {e}")

    file_index["last_incremental_update"] = datetime.now().isoformat()
    file_index["total_files_indexed"] = len(file_index["files"])
    file_index["total_chunks"] = vectorstore.get_stats()["total_chunks"]
    _save_file_index(index_path, file_index)

    elapsed = time.time() - start_time
    return {
        "new_files": new_files,
        "updated_files": updated_files,
        "deleted_files": deleted_files,
        "new_chunks": new_chunks,
        "total_chunks_after": vectorstore.get_stats()["total_chunks"],
        "time_ms": int(elapsed * 1000),
    }