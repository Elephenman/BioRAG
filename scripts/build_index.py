#!/usr/bin/env python
"""BioRAG 一键构建向量库"""

import sys
import os

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 确保能找到 biorag 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from biorag.config import load_config
from biorag.builder import build_index


def main():
    config_path = os.environ.get("BIORAG_CONFIG", os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    config = load_config(config_path)

    force = "--force" in sys.argv

    print("BioRAG 向量库构建")
    print(f"  知识库: {config.knowledge_base_path}")
    print(f"  模型: {config.embedding.model}")
    print(f"  模式: {'全量重建' if force else '增量更新'}")
    print()

    result = build_index(config, force=force)

    if "error" not in result:
        print(f"\n构建成功！")
        print(f"  总块数: {result.get('total_chunks_after', result.get('new_chunks', 'N/A'))}")

    return result


if __name__ == "__main__":
    main()