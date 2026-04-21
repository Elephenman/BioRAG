"""BioRAG Embedding 模型封装 - 本地 sentence-transformers"""

import os
from typing import List, Optional
from biorag.config import EmbeddingConfig


class BioRAGEmbedder:
    """Embedding 模型封装，支持本地模型"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        """延迟加载模型（首次使用时才加载，避免启动慢）"""
        if self._model is not None:
            return
        print(f"[WORK] 加载 Embedding 模型: {self.config.model}...")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            self.config.model,
            device=self.config.device,
            cache_folder=os.path.abspath(self.config.model_cache_dir),
        )
        print(f"[OK] 模型加载完成 (device={self.config.device})")

    def encode(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """编码文本列表，返回向量列表"""
        self._load_model()
        if batch_size is None:
            batch_size = self.config.batch_size
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> List[float]:
        """编码单条文本"""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @property
    def model_name(self) -> str:
        return self.config.model
