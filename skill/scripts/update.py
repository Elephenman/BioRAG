#!/usr/bin/env python
"""BioRAG 知识库增量更新"""

import argparse
import json
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import requests


def main():
    parser = argparse.ArgumentParser(description="BioRAG 知识库更新")
    parser.add_argument("--force", action="store_true", help="强制全量重建")
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()

    try:
        print(" 正在更新知识库...")
        resp = requests.post(
            f"{args.api_url}/update",
            json={"force_rebuild": args.force},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            print(f" 更新失败: {data['error']}")
        else:
            print(f" 更新完成！")
            print(f"  处理文件：{data['new_files']}")
            print(f"  新增块数：{data['new_chunks']}")
            print(f"  当前总块：{data['total_chunks_after']}")
            print(f"  耗时：{data['time_ms']/1000:.1f} 秒")

        print(f"\n--- JSON ---")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except requests.ConnectionError:
        print(json.dumps({"error": "Engine 未启动"}))
        sys.exit(1)


if __name__ == "__main__":
    main()