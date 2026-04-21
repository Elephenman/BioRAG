"""BioRAG 文档加载器 - 支持 md/R/pdf/ipynb"""

import os
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class LoadedDocument:
    """加载后的文档对象"""
    content: str
    source: str           # 完整文件路径
    filename: str         # 文件名
    file_type: str        # 扩展名
    folder: str           # 一级文件夹
    subfolder: str        # 二级文件夹
    file_size: int        # 字节数
    file_mtime: str       # 修改时间
    heading: str = ""     # 首个标题（md）


def _try_read(filepath: str, encodings: list = None) -> Optional[str]:
    """尝试多种编码读取文件"""
    if encodings is None:
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def _extract_first_heading(content: str) -> str:
    """提取 Markdown 文件的第一个标题"""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _get_folder_info(filepath: str, base_path: str) -> tuple:
    """获取一级和二级文件夹名"""
    rel = os.path.relpath(filepath, base_path)
    parts = rel.replace("\\", "/").split("/")
    folder = parts[0] if len(parts) > 1 else ""
    subfolder = parts[1] if len(parts) > 2 else ""
    return folder, subfolder


def _get_file_mtime(filepath: str) -> str:
    """获取文件修改时间 ISO 格式"""
    import datetime
    mtime = os.path.getmtime(filepath)
    return datetime.datetime.fromtimestamp(mtime).isoformat()


def load_markdown(filepath: str, base_path: str) -> Optional[LoadedDocument]:
    """加载 Markdown 文件"""
    content = _try_read(filepath)
    if content is None:
        return None
    folder, subfolder = _get_folder_info(filepath, base_path)
    return LoadedDocument(
        content=content,
        source=filepath,
        filename=os.path.basename(filepath),
        file_type="md",
        folder=folder,
        subfolder=subfolder,
        file_size=os.path.getsize(filepath),
        file_mtime=_get_file_mtime(filepath),
        heading=_extract_first_heading(content),
    )


def load_r_code(filepath: str, base_path: str) -> Optional[LoadedDocument]:
    """加载 R 代码文件"""
    content = _try_read(filepath)
    if content is None:
        return None
    folder, subfolder = _get_folder_info(filepath, base_path)
    # 提取 R 注释作为标题
    heading = ""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# ") and not line.startswith("# !"):
            heading = line.lstrip("# ").strip()
            break
    return LoadedDocument(
        content=content,
        source=filepath,
        filename=os.path.basename(filepath),
        file_type="R",
        folder=folder,
        subfolder=subfolder,
        file_size=os.path.getsize(filepath),
        file_mtime=_get_file_mtime(filepath),
        heading=heading,
    )


def load_pdf(filepath: str, base_path: str) -> Optional[LoadedDocument]:
    """加载 PDF 文件"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        content = "\n\n".join(pages)
        if not content.strip():
            return None
        folder, subfolder = _get_folder_info(filepath, base_path)
        return LoadedDocument(
            content=content,
            source=filepath,
            filename=os.path.basename(filepath),
            file_type="pdf",
            folder=folder,
            subfolder=subfolder,
            file_size=os.path.getsize(filepath),
            file_mtime=_get_file_mtime(filepath),
            heading="",
        )
    except Exception as e:
        print(f"  [WARN] PDF加载失败: {filepath} ({e})")
        return None


def load_ipynb(filepath: str, base_path: str) -> Optional[LoadedDocument]:
    """加载 Jupyter Notebook 文件"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = json.load(f)
        parts = []
        for cell in nb.get("cells", []):
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))
            if cell_type == "markdown" and source.strip():
                parts.append(f"[Markdown]\n{source}")
            elif cell_type == "code" and source.strip():
                parts.append(f"[Code]\n{source}")
        content = "\n\n".join(parts)
        if not content.strip():
            return None
        folder, subfolder = _get_folder_info(filepath, base_path)
        return LoadedDocument(
            content=content,
            source=filepath,
            filename=os.path.basename(filepath),
            file_type="ipynb",
            folder=folder,
            subfolder=subfolder,
            file_size=os.path.getsize(filepath),
            file_mtime=_get_file_mtime(filepath),
            heading="",
        )
    except Exception as e:
        print(f"  [WARN] ipynb加载失败: {filepath} ({e})")
        return None


# 文件类型 → 加载函数映射
LOADERS = {
    ".md": load_markdown,
    ".r": load_r_code,
    ".rmd": load_markdown,
    ".pdf": load_pdf,
    ".ipynb": load_ipynb,
    # R 代码大写扩展名
    ".rprofile": load_r_code,
}


def load_file(filepath: str, base_path: str) -> Optional[LoadedDocument]:
    """根据扩展名自动选择加载器"""
    ext = os.path.splitext(filepath)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        return None
    return loader(filepath, base_path)


def scan_knowledge_base(kb_path: str, ignore_patterns: list = None) -> List[str]:
    """扫描知识库目录，返回所有支持的文件路径列表"""
    if ignore_patterns is None:
        ignore_patterns = [".git", ".obsidian", ".trash"]

    files = []
    for root, dirs, filenames in os.walk(kb_path):
        dirs[:] = [d for d in dirs if d not in ignore_patterns and not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            if any(fn.endswith(pat.lstrip("*")) for pat in ignore_patterns if pat.startswith("*")):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in LOADERS:
                files.append(os.path.join(root, fn))

    return sorted(files)


def load_all_files(kb_path: str, ignore_patterns: list = None) -> List[LoadedDocument]:
    """扫描知识库目录，加载所有支持的文件"""
    if ignore_patterns is None:
        ignore_patterns = [".git", ".obsidian", ".trash"]

    docs = []
    skipped = 0
    failed = 0

    for root, dirs, files in os.walk(kb_path):
        # 过滤忽略的目录
        dirs[:] = [d for d in dirs if d not in ignore_patterns and not d.startswith(".")]

        for filename in files:
            # 过滤忽略的文件
            if any(filename.endswith(pat.lstrip("*")) for pat in ignore_patterns if pat.startswith("*")):
                skipped += 1
                continue
            if filename.startswith("."):
                skipped += 1
                continue

            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext not in LOADERS:
                skipped += 1
                continue

            doc = load_file(filepath, kb_path)
            if doc is not None:
                docs.append(doc)
            else:
                failed += 1

    print(f"[SCAN] 加载完成: {len(docs)} 个文件, 跳过 {skipped}, 失败 {failed}")
    return docs
