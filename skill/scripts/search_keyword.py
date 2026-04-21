#!/usr/bin/env python
"""BioRAG 关键词检索脚本"""

import argparse
import json
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import requests


def main():
    parser = argparse.ArgumentParser(description="BioRAG 关键词检索")
    parser.add_argument("keyword", help="关键词")
    parser.add_argument("--file-type", default=None, choices=["md", "R", "pdf"])
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    try:
        resp = requests.post(
            f"{args.api_url}/search_kw",
            json={
                "keyword": args.keyword,
                "file_type": args.file_type,
                "max_results": args.max_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    except requests.ConnectionError:
        print(json.dumps({"error": "Engine 未启动", "results": []}))
        sys.exit(1)


if __name__ == "__main__":
    main()