import os
import requests
import json
from typing import List, Dict
from tqdm import tqdm


def chunk_list(lst, chunk_size):
    if not isinstance(lst, list):
        raise TypeError("第一个参数必须是列表")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size必须是正整数")

    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def generate_results(data: List[Dict]) -> List[Dict]:
    """
    调用 RayRewardServer 的 HTTP 接口，拿回每个样本的打分结果。

    参数:
        data: List[dict]，每个元素形如：
            {
                "image": [...base64...],
                "question": str,
                "type": "single_choice"/"multi_choice"/"closed_ended",
                "choices": {...},
                "system": str,
            }

    返回:
        List[dict]，与 data 等长，每个元素至少包含:
            {
                "question": ...,
                "score": float (0~1),
                ...（例如 answer/results/responses 等）
            }
    """
    if not data:
        return []

    # 从环境变量中读取服务信息（由多机启动脚本注入）
    host = os.getenv("REWARD_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("REWARD_SERVICE_PORT", "8000"))
    handler_name = os.getenv("REWARD_HANDLER_NAME", "generate_questions")

    url = f"http://{host}:{port}/{handler_name}"

    payload = {
        "items": data
    }
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        resp_json = resp.json()
    except Exception as e:
        print(f"[reward] ERROR calling reward service: {e}")
        fallback = []
        for item in data:
            fallback.append(
                {
                    "question": item.get("question", ""),
                    "score": 0.0,
                    "error": f"reward service call failed: {repr(e)}",
                }
            )
        return fallback
    results = resp_json.get("results", [])
    return results


src_path = "/data/hanxiao36/projects/visplay_questioner_demo/wdm_batch1to6_sft_ZH_0_ner_fix_demo_train.json"
dst_path = "/data/hanxiao36/projects/visplay_questioner_demo/generate_questions.json"

with open(src_path, encoding="utf-8") as f:
    data = json.load(f)

chunk = 16  *  128

chunk_list = chunk_list(data, chunk)
total_results = []
for batch in tqdm(chunk_list):
    res = generate_results(batch)
    total_results.extend(res)

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(total_results, f, ensure_ascii=False, indent=4)
