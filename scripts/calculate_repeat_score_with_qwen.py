import json
import time
import torch
from vllm import LLM
from collections import Counter
from sklearn.cluster import AgglomerativeClustering


class Embedder:
    def __init__(self, embedding_model='Qwen/Qwen3-Embedding-0.6B'):
        self.embedding_model = LLM(model=embedding_model, task="embed")

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def compute_embeddings(self, texts, task_query=None, normalize=True):
        if task_query:
            texts = [self.get_detailed_instruct(task_query, t) for t in texts]

        outputs = self.embedding_model.embed(texts)
        emb = torch.tensor([o.outputs.embedding for o in outputs], dtype=torch.float32)

        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        return emb.cpu()


def cosine_distance_matrix(texts, task_query=None, symmetrize=True):
    """
    返回 NxN 距离矩阵，dist = 1 - cosine_sim
    - 对称
    - 对角线为 0
    """
    emb = embedder.compute_embeddings(texts, task_query=task_query, normalize=True)  # (N, D)

    # cosine_sim = emb @ emb.T because we L2-normalized
    sim = (emb @ emb.T).numpy()

    # 数值稳定：夹到 [-1, 1]
    sim = sim.clip(-1.0, 1.0)

    dist = 1.0 - sim  # (N, N)

    # 强制对角线为 0（避免数值误差）
    import numpy as np
    np.fill_diagonal(dist, 0.0)

    # 可选：再强制对称一次（理论上已对称，但保险）
    if symmetrize:
        dist = 0.5 * (dist + dist.T)

    return dist


def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.35,
        linkage: str = "average",
):
    if not problems:
        return []

    if linkage not in {"average", "complete", "single"}:
        raise ValueError("For precomputed distances, linkage must be one of: average, complete, single")

    print("start clustering")
    start_time = time.time()

    task_query = "Given a multiple-choice question query, retrieve another one whose content is most similar to the query"
    dist_mat = cosine_distance_matrix(problems, task_query=task_query)

    # sklearn 版本兼容：有的版本用 metric，有的用 affinity
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage=linkage,
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity="precomputed",
            linkage=linkage,
        )

    labels = clustering.fit_predict(dist_mat)

    print(f"end clustering, time: {time.time() - start_time:.2f}s")
    total = len(problems)

    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


def format_question(question, choices):
    options = []
    for idx, option in choices.items():
        options.append(f"{idx}: {option}")
    options = sorted(options, key=lambda x: x.split(":")[0])
    choices_str = ", ".join(options)
    return f"Question: {question}\n Choices: [{choices_str}]"


src_path = "/data/hanxiao36/projects/RayRewardServer/questions_results_with_scores.json"
dst_path = "/data/hanxiao36/projects/RayRewardServer/questions_results_with_scores_and_penalty_qwen.json"

embedder = Embedder("/data/share/250911/models/Qwen3-Embedding-0.6B")

with open(src_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# questions = [format_question(_["qa"]["question"], _["qa"]["choices"]) for _ in data]
questions = [_["qa"]["question"] for _ in data]

# 建议从 0.25~0.45 之间扫一遍看簇大小分布，这里先给个更保守的默认值
penalty = cluster_share_per_problem(questions, distance_threshold=0.35, linkage="average")

for i, item in enumerate(data):
    item["penalty"] = penalty[i]

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
