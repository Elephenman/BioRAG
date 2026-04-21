"""BioRAG FastAPI Engine 服务"""

import os
import sys
import time
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from biorag.config import load_config, BioRAGConfig, get_category_for_path
from biorag.embedder import BioRAGEmbedder
from biorag.vectorstore import BioRAGVectorStore
from biorag.metadata import MetadataDB
from biorag.search_log import SearchLogDB
from biorag.builder import build_index
from biorag.updater import incremental_update


# ===== 全局状态 =====
_config: Optional[BioRAGConfig] = None
_embedder: Optional[BioRAGEmbedder] = None
_vectorstore: Optional[BioRAGVectorStore] = None
_metadata_db: Optional[MetadataDB] = None
_search_log: Optional[SearchLogDB] = None
_start_time: float = 0


# ===== 请求模型 =====
class SearchRequest(BaseModel):
    query: str
    level: int = 1
    top_k: Optional[int] = None
    category: Optional[str] = None
    min_score: float = 0.3


class SearchKwRequest(BaseModel):
    keyword: str
    file_type: Optional[str] = None
    max_results: int = 10


class UpdateRequest(BaseModel):
    force_rebuild: bool = False


# ===== 应用创建 =====
app = FastAPI(title="BioRAG Engine", version="1.0.0")


@app.on_event("startup")
async def startup():
    global _config, _embedder, _vectorstore, _metadata_db, _search_log, _start_time
    _start_time = time.time()

    # 加载配置
    config_path = os.environ.get("BIORAG_CONFIG", "config.yaml")
    # 尝试多个路径
    for try_path in [config_path, os.path.join(os.path.dirname(__file__), "..", "config.yaml")]:
        if os.path.exists(try_path):
            _config = load_config(try_path)
            break
    if _config is None:
        _config = BioRAGConfig()
        print(f"[WARN] 未找到配置文件，使用默认配置")

    print(f"[LOG] 知识库路径: {_config.knowledge_base_path}")

    # 确保数据目录存在
    data_dir = os.path.abspath(_config.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # 加载 Embedding
    _embedder = BioRAGEmbedder(_config.embedding)

    # 连接向量库
    chroma_path = os.path.join(data_dir, "chroma_db")
    _vectorstore = BioRAGVectorStore(chroma_path, _embedder)

    # 连接元数据
    meta_path = os.path.join(data_dir, "metadata.db")
    _metadata_db = MetadataDB(meta_path)

    # 连接检索日志
    log_path = os.path.join(data_dir, "search_logs.db")
    _search_log = SearchLogDB(log_path)

    # 首次运行检查
    if _vectorstore.get_stats()["total_chunks"] == 0:
        print("[NEW] 首次运行，开始构建向量库...")
        build_index(_config, force=True)
        # Reinitialize vectorstore after build
        chroma_path = os.path.join(data_dir, "chroma_db")
        _vectorstore = BioRAGVectorStore(chroma_path, _embedder)
        _metadata_db = MetadataDB(meta_path)

    chunk_count = _vectorstore.get_stats()["total_chunks"]
    print(f"[OK] BioRAG Engine 启动完成！({chunk_count} 个文本块)")


@app.on_event("shutdown")
async def shutdown():
    if _metadata_db:
        _metadata_db.close()
    if _search_log:
        _search_log.close()
    print("BioRAG Engine 已停止")


# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== API 端点 =====

@app.get("/health")
async def health():
    return {"status": "running", "uptime_seconds": int(time.time() - _start_time)}


@app.post("/search")
async def search(req: SearchRequest):
    """语义检索"""
    if _vectorstore is None:
        raise HTTPException(503, "Engine 未就绪")

    start = time.time()

    # 确定 top_k 和截断参数
    level = req.level
    level_cfg = {
        1: _config.retrieval.level_1,
        2: _config.retrieval.level_2,
        3: _config.retrieval.level_3,
    }.get(level, _config.retrieval.level_1)

    top_k = req.top_k or level_cfg.top_k
    max_chars = level_cfg.max_chars_per_chunk

    # 检索
    results = _vectorstore.search(
        query=req.query,
        top_k=top_k,
        category=req.category,
        min_score=req.min_score,
    )

    # 截断内容 + 展平 metadata
    for r in results:
        if max_chars and level != 3 and len(r["content"]) > max_chars:
            r["content"] = r["content"][:max_chars] + "...(截断)"
        meta = r.pop("metadata", {})
        r["source"] = meta.get("filename", "")
        r["category"] = meta.get("category", "")
        r["heading"] = meta.get("heading", "")
        r["chunk_id"] = meta.get("chunk_id", 0)
        r["total_chunks"] = meta.get("total_chunks", 0)

    elapsed_ms = int((time.time() - start) * 1000)

    # 计算统计
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    min_s = min(scores) if scores else 0
    max_s = max(scores) if scores else 0

    # 记录日志
    sources = [r.get("source", "") for r in results]
    if _search_log:
        _search_log.log_search(
            query=req.query, level=level, category=req.category,
            top_k=top_k, results_count=len(results),
            avg_score=avg_score, min_score=min_s, max_score=max_s,
            time_ms=elapsed_ms, sources=sources,
        )

    return {
        "query": req.query,
        "level": level,
        "results": results,
        "total_searched": _vectorstore.get_stats()["total_chunks"],
        "time_ms": elapsed_ms,
        "stats": {
            "avg_score": round(avg_score, 4),
            "min_score": round(min_s, 4),
            "max_score": round(max_s, 4),
            "result_count": len(results),
        }
    }


@app.post("/search_kw")
async def search_keyword(req: SearchKwRequest):
    """关键词检索"""
    if _vectorstore is None:
        raise HTTPException(503, "Engine 未就绪")

    results = _vectorstore.search_keyword(
        keyword=req.keyword,
        file_type=req.file_type,
        max_results=req.max_results,
    )

    return {
        "keyword": req.keyword,
        "results": results,
        "total_matches": len(results),
    }


@app.get("/status")
async def status():
    """知识库状态"""
    if _metadata_db is None:
        raise HTTPException(503, "Engine 未就绪")

    total_stats = _metadata_db.get_total_stats()
    cat_stats = _metadata_db.get_category_stats()
    chunk_stats = _vectorstore.get_stats()

    # 计算数据库大小
    data_dir = os.path.abspath(_config.data_dir)
    db_size = 0
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            db_size += os.path.getsize(os.path.join(root, f))
    db_size_mb = db_size / (1024 * 1024)

    return {
        "total_documents": total_stats["total_documents"],
        "total_chunks": chunk_stats["total_chunks"],
        "db_size_mb": round(db_size_mb, 1),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_model": _config.embedding.model,
        "embedding_device": _config.embedding.device,
        "categories": cat_stats,
        "engine_status": "running",
        "uptime_seconds": int(time.time() - _start_time),
    }


@app.post("/update")
async def update(req: UpdateRequest):
    """增量更新"""
    if _config is None:
        raise HTTPException(503, "Engine 未就绪")

    result = incremental_update(_config, _vectorstore, _metadata_db, _embedder, force=req.force_rebuild)
    return result


@app.get("/stats/recent")
async def stats_recent(days: int = 7):
    """近期检索统计"""
    if _search_log is None:
        raise HTTPException(503, "Engine 未就绪")

    stats = _search_log.get_recent_stats(days)
    daily = _search_log.get_daily_stats(days)
    low_score = _search_log.get_low_score_queries(days)

    return {
        "period": f"{days}d",
        "total_searches": stats["total_searches"],
        "avg_similarity": stats["avg_similarity"],
        "low_score_searches": stats["low_score_count"],
        "avg_time_ms": stats["avg_time_ms"],
        "daily": daily,
        "low_score_queries": low_score,
    }


@app.get("/stats/hot")
async def stats_hot(days: int = 7, limit: int = 10):
    """热门文件"""
    if _search_log is None:
        raise HTTPException(503, "Engine 未就绪")

    hot_files = _search_log.get_hot_files(days, limit)
    return {"period": f"{days}d", "top_files": hot_files}


@app.get("/vectors/umap")
async def vectors_umap(n_neighbors: int = 15, min_dist: float = 0.1, refresh: bool = False):
    """UMAP 坐标导出"""
    if _vectorstore is None:
        raise HTTPException(503, "Engine 未就绪")

    # 检查缓存
    data_dir = os.path.abspath(_config.data_dir)
    cache_path = os.path.join(data_dir, "umap_cache.json")

    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return cache

    # 计算 UMAP
    try:
        from umap import UMAP
        import numpy as np
    except ImportError:
        raise HTTPException(500, "umap-learn 未安装，请运行: pip install umap-learn")

    all_data = _vectorstore.get_embeddings_for_umap()
    if not all_data["embeddings"]:
        return {"points": [], "params": {}}

    embeddings = np.array(all_data["embeddings"])
    metadatas = all_data["metadatas"]

    reducer = UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, metric="cosine", random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    points = []
    for i, (x, y) in enumerate(coords):
        meta = metadatas[i] if i < len(metadatas) else {}
        preview = all_data["documents"][i][:80] if i < len(all_data["documents"]) else ""
        points.append({
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "category": meta.get("category", "other"),
            "source": meta.get("filename", "unknown"),
            "heading": meta.get("heading", ""),
            "preview": preview + "..." if len(preview) > 80 else preview,
        })

    result = {
        "points": points,
        "params": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_points": len(points),
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    }

    # 缓存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    return result


# ===== 启动入口 =====
def main():
    import uvicorn
    config_path = os.environ.get("BIORAG_CONFIG", "config.yaml")
    global _config
    _config = load_config(config_path)
    uvicorn.run(app, host=_config.engine.host, port=_config.engine.port)


if __name__ == "__main__":
    main()