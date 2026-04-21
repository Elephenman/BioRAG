#!/usr/bin/env python
"""BioRAG 语义检索脚本"""

import argparse
import json
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import requests


def main():
    parser = argparse.ArgumentParser(description="BioRAG 知识检索")
    parser.add_argument("query", help="检索查询")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--category", default=None,
                       choices=["bioinfo", "r_code", "ml", "other"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--full-doc", action="store_true", help="Level 3 返回完整文档")
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    try:
        resp = requests.post(
            f"{args.api_url}/search",
            json={
                "query": args.query,
                "level": args.level,
                "top_k": args.top_k,
                "category": args.category,
                "min_score": args.min_score,
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.ConnectionError:
        print(json.dumps({
            "error": "BioRAG Engine 未启动，请先运行: python -m biorag.engine",
            "results": [],
        }))
        sys.exit(1)
    except requests.Timeout:
        print(json.dumps({"error": "请求超时", "results": []}))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()