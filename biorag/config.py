"""BioRAG 配置加载模块

使用 dataclass 从 config.yaml 加载配置，缺失字段用默认值填充。
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    provider: str = "local"
    model: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"
    dimension: int = 512
    batch_size: int = 32
    model_cache_dir: str = "./models"


@dataclass
class ChunkTypeConfig:
    """单个分块类型配置，包含可选的 extract_headings / preserve_functions / extract_code_cells / extract_markdown_cells"""
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " "])
    extract_headings: bool = False
    preserve_functions: bool = False
    extract_code_cells: bool = False
    extract_markdown_cells: bool = False


@dataclass
class ChunkingConfig:
    """分块配置汇总"""
    md: ChunkTypeConfig = field(default_factory=ChunkTypeConfig)
    r_code: ChunkTypeConfig = field(default_factory=lambda: ChunkTypeConfig(
        chunk_size=800, chunk_overlap=150, separators=["\n# ", "\n\n", "\n"],
        preserve_functions=True
    ))
    pdf: ChunkTypeConfig = field(default_factory=ChunkTypeConfig)
    ipynb: ChunkTypeConfig = field(default_factory=lambda: ChunkTypeConfig(
        chunk_size=800, chunk_overlap=150, extract_code_cells=True, extract_markdown_cells=True
    ))


@dataclass
class RetrievalLevelConfig:
    """检索级别配置"""
    top_k: int = 3
    max_chars_per_chunk: Optional[int] = 200
    max_total_chars: Optional[int] = 600


@dataclass
class RetrievalConfig:
    """检索配置汇总，含 level_1 / level_2 / level_3"""
    level_1: RetrievalLevelConfig = field(default_factory=lambda: RetrievalLevelConfig(3, 200, 600))
    level_2: RetrievalLevelConfig = field(default_factory=lambda: RetrievalLevelConfig(8, 300, 2500))
    level_3: RetrievalLevelConfig = field(default_factory=lambda: RetrievalLevelConfig(5, None, None))
    min_similarity: float = 0.3
    default_category: Optional[str] = None


@dataclass
class EngineConfig:
    """Engine 服务配置"""
    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:*", "tauri://localhost"])
    log_level: str = "INFO"
    auto_start: bool = True


@dataclass
class UmapConfig:
    """UMAP 可视化配置"""
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    cache_enabled: bool = True
    cache_max_age_hours: int = 24


@dataclass
class VisualizationConfig:
    """可视化配置汇总"""
    umap: UmapConfig = field(default_factory=UmapConfig)


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    file: str = "./data/biorag.log"
    max_size_mb: int = 50
    backup_count: int = 3


@dataclass
class BioRAGConfig:
    """BioRAG 主配置"""
    knowledge_base_path: str = ""
    knowledge_base_watch: bool = False
    knowledge_base_ignore: List[str] = field(default_factory=lambda: [".git", ".obsidian", ".trash", "*.tmp", "*.bak", "*/.*"])
    data_dir: str = "./data"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    categories: Dict[str, List[str]] = field(default_factory=dict)
    engine: EngineConfig = field(default_factory=EngineConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ── 内部解析辅助 ──────────────────────────────────────────


def _parse_chunk_type(raw: dict, defaults: ChunkTypeConfig) -> ChunkTypeConfig:
    """解析单个分块类型配置"""
    if not raw:
        return defaults
    return ChunkTypeConfig(
        chunk_size=raw.get("chunk_size", defaults.chunk_size),
        chunk_overlap=raw.get("chunk_overlap", defaults.chunk_overlap),
        separators=raw.get("separators", defaults.separators),
        extract_headings=raw.get("extract_headings", defaults.extract_headings),
        preserve_functions=raw.get("preserve_functions", defaults.preserve_functions),
        extract_code_cells=raw.get("extract_code_cells", defaults.extract_code_cells),
        extract_markdown_cells=raw.get("extract_markdown_cells", defaults.extract_markdown_cells),
    )


def _parse_retrieval_level(raw: dict, defaults: RetrievalLevelConfig) -> RetrievalLevelConfig:
    """解析检索级别配置"""
    if not raw:
        return defaults
    return RetrievalLevelConfig(
        top_k=raw.get("top_k", defaults.top_k),
        max_chars_per_chunk=raw.get("max_chars_per_chunk", defaults.max_chars_per_chunk),
        max_total_chars=raw.get("max_total_chars", defaults.max_total_chars),
    )


# ── 公共接口 ──────────────────────────────────────────────


def load_config(path: str = "config.yaml") -> BioRAGConfig:
    """从 YAML 文件加载配置，缺失字段用默认值填充"""
    config = BioRAGConfig()

    if not os.path.exists(path):
        print(f"[WARN] 配置文件不存在: {path}，使用默认配置")
        return config

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 知识库
    kb = raw.get("knowledge_base", {})
    config.knowledge_base_path = kb.get("path", "")
    config.knowledge_base_watch = kb.get("watch", config.knowledge_base_watch)
    config.knowledge_base_ignore = kb.get("ignore", config.knowledge_base_ignore)

    # 数据目录
    data = raw.get("data", {})
    config.data_dir = data.get("dir", config.data_dir)

    # Embedding
    emb = raw.get("embedding", {})
    config.embedding = EmbeddingConfig(
        provider=emb.get("provider", config.embedding.provider),
        model=emb.get("model", config.embedding.model),
        device=emb.get("device", config.embedding.device),
        dimension=emb.get("dimension", config.embedding.dimension),
        batch_size=emb.get("batch_size", config.embedding.batch_size),
        model_cache_dir=emb.get("model_cache_dir", config.embedding.model_cache_dir),
    )

    # 分块
    chk = raw.get("chunking", {})
    config.chunking = ChunkingConfig(
        md=_parse_chunk_type(chk.get("md", {}), config.chunking.md),
        r_code=_parse_chunk_type(chk.get("r_code", {}), config.chunking.r_code),
        pdf=_parse_chunk_type(chk.get("pdf", {}), config.chunking.pdf),
        ipynb=_parse_chunk_type(chk.get("ipynb", {}), config.chunking.ipynb),
    )

    # 检索
    ret = raw.get("retrieval", {})
    config.retrieval = RetrievalConfig(
        level_1=_parse_retrieval_level(ret.get("level_1", {}), config.retrieval.level_1),
        level_2=_parse_retrieval_level(ret.get("level_2", {}), config.retrieval.level_2),
        level_3=_parse_retrieval_level(ret.get("level_3", {}), config.retrieval.level_3),
        min_similarity=ret.get("min_similarity", config.retrieval.min_similarity),
        default_category=ret.get("default_category", config.retrieval.default_category),
    )

    # 分类映射
    config.categories = raw.get("categories", config.categories)

    # Engine
    eng = raw.get("engine", {})
    config.engine = EngineConfig(
        host=eng.get("host", config.engine.host),
        port=eng.get("port", config.engine.port),
        cors_origins=eng.get("cors_origins", config.engine.cors_origins),
        log_level=eng.get("log_level", config.engine.log_level),
        auto_start=eng.get("auto_start", config.engine.auto_start),
    )

    # 可视化
    vis = raw.get("visualization", {})
    umap_raw = vis.get("umap", {})
    config.visualization = VisualizationConfig(
        umap=UmapConfig(
            n_neighbors=umap_raw.get("n_neighbors", 15),
            min_dist=umap_raw.get("min_dist", 0.1),
            metric=umap_raw.get("metric", "cosine"),
            cache_enabled=umap_raw.get("cache_enabled", True),
            cache_max_age_hours=umap_raw.get("cache_max_age_hours", 24),
        )
    )

    # 日志
    log = raw.get("logging", {})
    config.logging = LoggingConfig(
        level=log.get("level", config.logging.level),
        file=log.get("file", config.logging.file),
        max_size_mb=log.get("max_size_mb", config.logging.max_size_mb),
        backup_count=log.get("backup_count", config.logging.backup_count),
    )

    return config


def get_category_for_path(file_path: str, categories: Dict[str, List[str]]) -> str:
    """根据文件路径判断所属分类"""
    for category, folders in categories.items():
        for folder in folders:
            if folder in file_path:
                return category
    return "other"