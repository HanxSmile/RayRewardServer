#!/usr/bin/env python3
# send_batch.py
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union

import requests


def load_payload(json_path: str) -> Dict[str, Any]:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with p.open("r", encoding="utf-8") as f:
        data: Union[Dict[str, Any], Any] = json.load(f)

    # 兼容两种格式：
    # 1) 直接是一个 batch 列表 -> {"items": [...]}
    # 2) 已经是 {"items": [...]} -> 原样
    if isinstance(data, list):
        return {"items": data}
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data

    raise ValueError(
        "Unsupported JSON format. Expect either:\n"
        "  - a list: [ {...}, {...} ]\n"
        "  - or a dict with items: {\"items\": [ ... ]}\n"
        f"Got: {type(data).__name__}"
    )


def main():
    parser = argparse.ArgumentParser(description="Read a JSON batch file and POST to handler endpoint.")
    parser.add_argument("--json", required=True, help="Path to batch json file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Service base URL")
    parser.add_argument("--handler", required=True, help="Handler name, e.g. questioner / dummy / vllm_chat")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout seconds")
    parser.add_argument("--out", default=None, help="Optional path to save response json")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print response")
    args = parser.parse_args()

    payload = load_payload(args.json)
    url = args.base_url.rstrip("/") + f"/{args.handler}"

    try:
        r = requests.post(url, json=payload, timeout=args.timeout)
    except requests.RequestException as e:
        print(f"[ERROR] Request failed: {e}", file=sys.stderr)
        sys.exit(2)

    # 尽量解析 JSON，否则打印文本
    content_type = r.headers.get("content-type", "")
    is_json = "application/json" in content_type.lower()

    print(f"POST {url}")
    print(f"Status: {r.status_code}")

    if is_json:
        resp = r.json()
        if args.pretty:
            print(json.dumps(resp, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(resp, ensure_ascii=False))
        if args.out:
            Path(args.out).write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved response to: {args.out}")
    else:
        print(r.text)
        if args.out:
            Path(args.out).write_text(r.text, encoding="utf-8")
            print(f"Saved response to: {args.out}")

    # 非 2xx 认为失败（但仍然把 body 打出来/保存出来，便于排错）
    if not (200 <= r.status_code < 300):
        sys.exit(1)


if __name__ == "__main__":
    main()
