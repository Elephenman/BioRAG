"""BioRAG 向量库构建器"""

import hashlib
import os
import json
import time
from typing import List

from biorag.config import BioRAGConfig, get_category_for_path
from biorag.loader import load_all_files, LoadedDocument
from biorag.chunker import chunk_all_documents, Chunk
from biorag.embedder import BioRAGEmbedder
from biorag.vectorstore import BioRAGVectorStore
from biorag.metadata import MetadataDB


def compute_file_hash(filepath: str) -> str:
    """计算文件 SHA256 哈希"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_file_index(index_path: str) -> dict:
    """加载文件索引"""
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": "2.0", "files": {}}


def save_file_index(index: dict, index_path: str):
    """保存文件索引"""
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def build_index(config: BioRAGConfig, force: bool = False) -> dict:
    """构建向量库
    
    Args:
        config: 配置对象
        force: 是否强制全量重建
    
    Returns:
        构建结果统计
    """
    start_time = time.time()
    
    # 确保数据目录存在
    data_dir = os.path.abspath(config.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    chroma_path = os.path.join(data_dir, "chroma_db")
    meta_path = os.path.join(data_dir, "metadata.db")
    index_path = os.path.join(data_dir, "file_index.json")
    
    # 1. 加载 Embedding 模型
    embedder = BioRAGEmbedder(config.embedding)
    
    # 2. 初始化向量库
    vectorstore = BioRAGVectorStore(chroma_path, embedder)
    
    # 3. 初始化元数据
    metadata_db = MetadataDB(meta_path)
    
    # 4. 加载文件索引
    file_index = load_file_index(index_path)
    
    # 5. 扫描知识库文件
    print(f"[SCAN] 扫描知识库: {config.knowledge_base_path}")
    docs = load_all_files(config.knowledge_base_path, config.knowledge_base_ignore)
    
    if not docs:
        print("[WARN] 未找到任何文件")
        return {"error": "no files found"}
    
    # 6. 计算哪些文件需要处理
    to_process = []
    if force:
        to_process = docs
        print(f"[WORK] 全量重建: {len(docs)} 个文件")
    else:
        existing_hashes = metadata_db.get_all_hashes()
        for doc in docs:
            file_hash = compute_file_hash(doc.source)
            old_hash = existing_hashes.get(doc.source)
            if old_hash != file_hash:
                to_process.append(doc)
        print(f"[WORK] 增量处理: {len(to_process)} 个文件需要处理 (共 {len(docs)} 个)")
    
    if not to_process and not force:
        print("[OK] 知识库无变化，无需更新")
        metadata_db.close()
        return {"new_files": 0, "updated_files": 0, "new_chunks": 0}
    
    # 7. 删除需要更新的旧数据
    for doc in to_process:
        vectorstore.delete_by_source(doc.source)
        metadata_db.delete_file(doc.source)
    
    # 8. 分块
    print("[CUT] 分块处理...")
    chunks = chunk_all_documents(to_process, config.chunking)
    
    # 9. 填充 category
    category_func = lambda source: get_category_for_path(source, config.categories)
    for chunk in chunks:
        if not chunk.metadata.get("category"):
            chunk.metadata["category"] = category_func(chunk.metadata.get("source", ""))
    
    # 10. 写入向量库
    print("[SAVE] 写入向量库...")
    vectorstore.add_documents(chunks, category_func)
    
    # 11. 更新元数据
    print("[META] 更新元数据...")
    for doc in to_process:
        file_hash = compute_file_hash(doc.source)
        doc_chunks = [c for c in chunks if c.metadata.get("source") == doc.source]
        category = category_func(doc.source)
        metadata_db.upsert_file(
            source=doc.source,
            filename=doc.filename,
            category=category,
            file_type=doc.file_type,
            folder=doc.folder,
            subfolder=doc.subfolder,
            file_size=doc.file_size,
            file_mtime=doc.file_mtime,
            chunk_count=len(doc_chunks),
            char_count=sum(c.metadata.get("char_count", 0) for c in doc_chunks),
            file_hash=file_hash,
        )
    
    # 12. 更新文件索引
    for doc in to_process:
        file_hash = compute_file_hash(doc.source)
        file_index["files"][doc.source] = {
            "hash": file_hash,
            "mtime": doc.file_mtime,
            "chunk_count": len([c for c in chunks if c.metadata.get("source") == doc.source]),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "indexed",
        }
    file_index["last_incremental_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    file_index["total_files_indexed"] = len(file_index["files"])
    file_index["embedding_model"] = config.embedding.model
    save_file_index(file_index, index_path)
    
    # 13. 统计
    elapsed = time.time() - start_time
    stats = metadata_db.get_total_stats()
    result = {
        "new_files": len(to_process),
        "updated_files": 0,
        "new_chunks": len(chunks),
        "total_chunks_after": stats["total_chunks"],
        "time_ms": int(elapsed * 1000),
    }
    
    print(f"[OK] 构建完成！")
    print(f"   处理文件: {result['new_files']}")
    print(f"   新增块数: {result['new_chunks']}")
    print(f"   当前总块: {result['total_chunks_after']}")
    print(f"   耗时: {elapsed:.1f} 秒")
    
    metadata_db.close()
    return result


def update_index(config: BioRAGConfig) -> dict:
    """增量更新"""
    return build_index(config, force=False)