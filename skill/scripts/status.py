#!/usr/bin/env python
"""BioRAG 知识库状态查询"""

import json
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import requests


def main():
    try:
        resp = requests.get("http://127.0.0.1:8765/status", timeout=5)
        resp.raise_for_status()
        data = resp.json()

        print("BioRAG 知识库状态")
        print("---------------------")
        print(f"文档总数：{data['total_documents']}")
        print(f"文本块数：{data['total_chunks']}")
        print(f"数据库大小：{data['db_size_mb']:.1f} MB")
        print(f"最后更新：{data['last_updated']}")
        print(f"Embedding：{data['embedding_model']} ({data['embedding_device']})")
        print(f"\n分类统计：")
        for cat, info in data["categories"].items():
            print(f"  {cat}: {info['file_count']} 文件, {info['chunk_count']} 块")

        print(f"\n--- JSON ---")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except requests.ConnectionError:
        print(json.dumps({"error": "Engine 未启动，请先运行: python -m biorag.engine"}))
        sys.exit(1)


if __name__ == "__main__":
    main()