"""BioRAG 文档分块器 - 按文件类型分块"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
from biorag.loader import LoadedDocument
from biorag.config import ChunkingConfig, ChunkTypeConfig


@dataclass
class Chunk:
    """分块结果"""
    content: str
    metadata: Dict[str, Any]


def _split_by_separators(text: str, separators: list, chunk_size: int, chunk_overlap: int) -> List[str]:
    """按分隔符递归切分文本"""
    if len(text) <= chunk_size:
        return [text]

    # 尝试每个分隔符
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    # 如果单个 part 超过 chunk_size，继续切
                    if len(part) > chunk_size:
                        # 按字符硬切
                        for i in range(0, len(part), chunk_size - chunk_overlap):
                            sub = part[i:i + chunk_size]
                            if sub.strip():
                                chunks.append(sub.strip())
                    current = part
            if current.strip():
                chunks.append(current.strip())
            return chunks

    # 没有分隔符能切，按字符硬切
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        sub = text[i:i + chunk_size]
        if sub.strip():
            chunks.append(sub.strip())
    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """为相邻块添加重叠内容"""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            prev_tail = chunks[i - 1][-overlap:]
            result.append(prev_tail + chunk)
        else:
            result.append(chunk)
    return result


def _extract_headings_md(content: str) -> str:
    """提取 Markdown 当前块所属的标题层级"""
    lines = content.strip().split("\n")
    for line in reversed(lines):
        if line.strip().startswith("#"):
            return line.strip().lstrip("#").strip()
    return ""


def chunk_markdown(doc: LoadedDocument, cfg: ChunkTypeConfig) -> List[Chunk]:
    """Markdown 文件分块"""
    raw_chunks = _split_by_separators(doc.content, cfg.separators, cfg.chunk_size, cfg.chunk_overlap)
    chunks = []
    for i, text in enumerate(raw_chunks):
        if not text.strip():
            continue
        heading = _extract_headings_md(text) if cfg.extract_headings else doc.heading
        chunks.append(Chunk(
            content=text,
            metadata={
                "source": doc.source,
                "filename": doc.filename,
                "category": "",  # 由外部填充
                "file_type": doc.file_type,
                "chunk_id": i,
                "total_chunks": len(raw_chunks),
                "heading": heading or doc.heading,
                "folder": doc.folder,
                "subfolder": doc.subfolder,
                "char_count": len(text),
            }
        ))
    return chunks


def _split_r_by_functions(content: str) -> List[str]:
    """尽量按 R 函数分割"""
    # 匹配函数定义
    func_pattern = re.compile(r'^(?:\w+::)?\w+\s*<?-\s*function\s*\(', re.MULTILINE)
    positions = list(func_pattern.finditer(content))

    if not positions:
        return [content]

    parts = []
    prev_end = 0
    for match in positions:
        if match.start() > prev_end:
            # 函数前的注释/代码
            parts.append(content[prev_end:match.start()])
        prev_end = match.start()

    # 最后一段
    if prev_end < len(content):
        parts.append(content[prev_end:])

    return [p for p in parts if p.strip()]


def chunk_r_code(doc: LoadedDocument, cfg: ChunkTypeConfig) -> List[Chunk]:
    """R 代码分块"""
    if cfg.preserve_functions:
        # 先按函数分割，再按大小限制
        func_parts = _split_r_by_functions(doc.content)
        raw_chunks = []
        for part in func_parts:
            if len(part) <= cfg.chunk_size:
                raw_chunks.append(part)
            else:
                raw_chunks.extend(_split_by_separators(part, cfg.separators, cfg.chunk_size, cfg.chunk_overlap))
    else:
        raw_chunks = _split_by_separators(doc.content, cfg.separators, cfg.chunk_size, cfg.chunk_overlap)

    chunks = []
    for i, text in enumerate(raw_chunks):
        if not text.strip():
            continue
        chunks.append(Chunk(
            content=text,
            metadata={
                "source": doc.source,
                "filename": doc.filename,
                "category": "",
                "file_type": doc.file_type,
                "chunk_id": i,
                "total_chunks": len(raw_chunks),
                "heading": doc.heading,
                "folder": doc.folder,
                "subfolder": doc.subfolder,
                "char_count": len(text),
            }
        ))
    return chunks


def chunk_pdf(doc: LoadedDocument, cfg: ChunkTypeConfig) -> List[Chunk]:
    """PDF 分块"""
    raw_chunks = _split_by_separators(doc.content, cfg.separators, cfg.chunk_size, cfg.chunk_overlap)
    chunks = []
    for i, text in enumerate(raw_chunks):
        if not text.strip():
            continue
        chunks.append(Chunk(
            content=text,
            metadata={
                "source": doc.source,
                "filename": doc.filename,
                "category": "",
                "file_type": doc.file_type,
                "chunk_id": i,
                "total_chunks": len(raw_chunks),
                "heading": "",
                "folder": doc.folder,
                "subfolder": doc.subfolder,
                "char_count": len(text),
            }
        ))
    return chunks


def chunk_ipynb(doc: LoadedDocument, cfg: ChunkTypeConfig) -> List[Chunk]:
    """Jupyter Notebook 分块"""
    # ipynb 在 loader 阶段已按 cell 拆分并标注
    raw_chunks = _split_by_separators(doc.content, cfg.separators, cfg.chunk_size, cfg.chunk_overlap)
    chunks = []
    for i, text in enumerate(raw_chunks):
        if not text.strip():
            continue
        chunks.append(Chunk(
            content=text,
            metadata={
                "source": doc.source,
                "filename": doc.filename,
                "category": "",
                "file_type": doc.file_type,
                "chunk_id": i,
                "total_chunks": len(raw_chunks),
                "heading": "",
                "folder": doc.folder,
                "subfolder": doc.subfolder,
                "char_count": len(text),
            }
        ))
    return chunks


# 文件类型 → 分块函数映射
CHUNKERS = {
    "md": chunk_markdown,
    "R": chunk_r_code,
    "r": chunk_r_code,
    "pdf": chunk_pdf,
    "ipynb": chunk_ipynb,
}


def chunk_document(doc: LoadedDocument, chunking_config: ChunkingConfig) -> List[Chunk]:
    """根据文件类型选择分块器"""
    # 选择对应的分块配置
    type_configs = {
        "md": chunking_config.md,
        "R": chunking_config.r_code,
        "r": chunking_config.r_code,
        "pdf": chunking_config.pdf,
        "ipynb": chunking_config.ipynb,
    }
    cfg = type_configs.get(doc.file_type, chunking_config.md)
    chunker = CHUNKERS.get(doc.file_type, chunk_markdown)
    return chunker(doc, cfg)


def chunk_all_documents(docs: List[LoadedDocument], chunking_config: ChunkingConfig) -> List[Chunk]:
    """批量分块"""
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunking_config)
        all_chunks.extend(chunks)
    print(f"[CUT] 分块完成: {len(docs)} 文档 → {len(all_chunks)} 个文本块")
    return all_chunks
