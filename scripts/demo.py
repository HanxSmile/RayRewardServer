import json
from typing import List, Dict


def generate_results(data: List[Dict]) -> List[Dict]:
    if not data:
        return []
    data_ = []
    raw_data = []
    for item in data:
        if item["qa"] is None:
            continue
        res = {
            "question": item["qa"]["question"],
            "type": item["question_type"],
            "choices": item["qa"]["choices"],
            "system": item["sys_prompt"],
            "image": item["images"],
        }
        data_.append(res)
        raw_data.append(item)

    results = model.process_batch(data_)

    total_results = []
    for result, item in zip(results, raw_data):
        if result["score"] >= 0.0:
            item["qa_res"] = result
            total_results.append(item)
    return total_results


from handlers.questioner import QuestionerHandler

model = QuestionerHandler(
    model_path="/data/hanxiao36/models/checkpoint-2000",
    tensor_parallel_size=1,
    max_model_len=20480,
    gpu_mem_util=0.9,
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    num_samples=10,
)

src_path = "/data/hanxiao36/projects/RayRewardServer/questions_results.json"
dst_path = "/data/hanxiao36/projects/RayRewardServer/questions_results_with_scores.json"

with open(src_path, encoding="utf-8") as f:
    data = json.load(f)

chunk = 16 * 128

total_results = generate_results(data)

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(total_results, f, ensure_ascii=False, indent=4)
