"""BioRAG ChromaDB 向量存储封装"""

import os
from typing import List, Dict, Any, Optional
import chromadb


class BioRAGVectorStore:
    """ChromaDB 操作封装"""

    def __init__(self, persist_dir: str, embedder, collection_name: str = "biorag_knowledge"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder

    def add_documents(self, chunks: list, category_func=None) -> int:
        """批量添加文档块"""
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        texts_to_embed = []

        for chunk in chunks:
            # 填充分类
            if category_func and not chunk.metadata.get("category"):
                chunk.metadata["category"] = category_func(chunk.metadata.get("source", ""))

            chunk_id = f"{chunk.metadata['source']}__{chunk.metadata['chunk_id']}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)
            texts_to_embed.append(chunk.content)

        # 批量编码
        print(f"  [WORK] 编码 {len(texts_to_embed)} 个文本块...")
        embeddings = self.embedder.encode(texts_to_embed)

        # 分批写入 ChromaDB（每批500个，避免内存溢出）
        batch_size = 500
        total_added = 0
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end],
            )
            total_added += end - i
            print(f"  [PACK] 写入 {min(end, len(ids))}/{len(ids)}")

        return total_added

    def search(self, query: str, top_k: int = 5,
               category: str = None, min_score: float = 0.3) -> List[Dict]:
        """语义检索"""
        query_embedding = self.embedder.encode_single(query)

        where_filter = None
        if category:
            where_filter = {"category": category}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"  [WARN] 检索失败: {e}")
            return []

        if not results["documents"] or not results["documents"][0]:
            return []

        # 过滤低分结果
        filtered = []
        for i, doc in enumerate(results["documents"][0]):
            score = 1 - results["distances"][0][i]  # cosine distance → similarity
            if score >= min_score:
                filtered.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "score": round(score, 4),
                })

        return filtered

    def search_keyword(self, keyword: str, file_type: str = None,
                       max_results: int = 10) -> List[Dict]:
        """关键词检索（ChromaDB 的 where document 过滤）"""
        where_filter = {"$contains": keyword}
        if file_type:
            where_filter = {"$and": [
                {"file_type": file_type},
                {"document": {"$contains": keyword}}
            ]}

        try:
            results = self.collection.query(
                query_embeddings=None,
                where=where_filter if isinstance(where_filter, dict) and "document" not in where_filter else None,
                where_document={"$contains": keyword},
                n_results=max_results,
                include=["documents", "metadatas"]
            )
        except Exception as e:
            print(f"  [WARN] 关键词检索失败: {e}")
            return []

        if not results["documents"] or not results["documents"][0]:
            return []

        items = []
        for i, doc in enumerate(results["documents"][0]):
            items.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "match_type": "content",
            })
        return items

    def delete_by_source(self, source: str) -> int:
        """按文件路径删除所有相关 chunk"""
        try:
            self.collection.delete(where={"source": source})
            return 1
        except Exception:
            return 0

    def get_all_sources(self) -> set:
        """获取所有已入库的文件路径"""
        try:
            all_meta = self.collection.get(include=["metadatas"])
            sources = set()
            for meta in all_meta["metadatas"]:
                if "source" in meta:
                    sources.add(meta["source"])
            return sources
        except Exception:
            return set()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_chunks": self.collection.count(),
        }

    def get_embeddings_for_umap(self) -> tuple:
        """获取所有向量和元数据，供 UMAP 使用"""
        all_data = self.collection.get(include=["embeddings", "metadatas", "documents"])
        return all_data
