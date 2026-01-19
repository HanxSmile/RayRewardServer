import json
import time
import torch
from vllm import LLM
from collections import Counter, defaultdict
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
    import numpy as np

    emb = embedder.compute_embeddings(texts, task_query=task_query, normalize=True)  # (N, D)
    sim = (emb @ emb.T).numpy()  # cosine_sim because we normalized
    sim = sim.clip(-1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    if symmetrize:
        dist = 0.5 * (dist + dist.T)
    return dist


def _print_threshold_diagnostics(dist_mat, labels, top_k=10, sample_top_clusters=5, max_pairs_per_cluster=200, seed=42):
    """
    打印：簇分布、最大簇占比、单例占比、分桶统计、以及簇内距离抽样（帮助判断阈值松/严）
    """
    import numpy as np

    n = len(labels)
    cluster_size = Counter(labels)
    sizes = np.array(list(cluster_size.values()), dtype=int)

    n_clusters = len(sizes)
    max_sz = int(sizes.max()) if n > 0 else 0
    singleton_clusters = int((sizes == 1).sum())
    singleton_points = singleton_clusters  # 每个单例簇就是1个点
    max_ratio = max_sz / n if n else 0.0
    singleton_point_ratio = singleton_points / n if n else 0.0

    print("\n========== distance_threshold diagnostics ==========")
    print(f"Total points: {n}")
    print(f"Num clusters: {n_clusters}")
    print(f"Largest cluster size: {max_sz} ({max_ratio:.2%} of all points)")
    print(f"Singleton points: {singleton_points} ({singleton_point_ratio:.2%} of all points)")

    # 分桶统计：看整体粒度
    bins = {
        "1": 0,
        "2": 0,
        "3-5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-50": 0,
        "51+": 0,
    }
    for sz in sizes:
        if sz == 1:
            bins["1"] += 1
        elif sz == 2:
            bins["2"] += 1
        elif 3 <= sz <= 5:
            bins["3-5"] += 1
        elif 6 <= sz <= 10:
            bins["6-10"] += 1
        elif 11 <= sz <= 20:
            bins["11-20"] += 1
        elif 21 <= sz <= 50:
            bins["21-50"] += 1
        else:
            bins["51+"] += 1

    print("\nCluster count by size bucket (counts are #clusters):")
    for k, v in bins.items():
        print(f"  size {k:>4}: {v}")

    # Top-K 最大簇
    top = sorted(cluster_size.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"\nTop-{top_k} largest clusters:")
    for lab, sz in top:
        print(f"  label={lab:<6} size={sz:<6} ratio={sz / n:.2%}")

    # 簇内距离抽样：看同簇内部“到底像不像”
    # 只对最大的几个簇做，避免太慢
    if sample_top_clusters > 0 and max_pairs_per_cluster > 0:
        rng = np.random.default_rng(seed)
        print("\nIntra-cluster distance sampling (mean/median/p90; lower is more similar):")
        for lab, sz in top[:sample_top_clusters]:
            if sz < 2:
                continue
            idx = np.where(labels == lab)[0]
            m = len(idx)

            # 抽样 pairs
            num_pairs = min(max_pairs_per_cluster, m * (m - 1) // 2)
            if num_pairs <= 0:
                continue

            # 生成随机 pair（去掉 i==j）
            # 简化：随机抽两列索引并过滤 i!=j；数量不够就多抽几轮
            pairs_i = []
            pairs_j = []
            tries = 0
            while len(pairs_i) < num_pairs and tries < 10:
                need = num_pairs - len(pairs_i)
                ii = rng.integers(0, m, size=need * 2)
                jj = rng.integers(0, m, size=need * 2)
                mask = ii != jj
                ii = ii[mask]
                jj = jj[mask]
                take = min(len(ii), need)
                pairs_i.extend(ii[:take].tolist())
                pairs_j.extend(jj[:take].tolist())
                tries += 1

            ii = idx[np.array(pairs_i, dtype=int)]
            jj = idx[np.array(pairs_j, dtype=int)]
            d = dist_mat[ii, jj]

            d_mean = float(np.mean(d))
            d_median = float(np.median(d))
            d_p90 = float(np.quantile(d, 0.90))
            print(
                f"  label={lab:<6} size={sz:<6} sample_pairs={len(d):<4} mean={d_mean:.4f} median={d_median:.4f} p90={d_p90:.4f}")

    print("===================================================\n")


def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.35,
        linkage: str = "average",
        diagnostics: bool = True,
):
    if not problems:
        return []

    if linkage not in {"average", "complete", "single"}:
        raise ValueError("For precomputed distances, linkage must be one of: average, complete, single")

    print("start clustering")
    start_time = time.time()

    task_query = "Given a medical related question query, retrieve another one whose content is most similar to the query"
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

    if diagnostics:
        _print_threshold_diagnostics(dist_mat, labels, top_k=10, sample_top_clusters=5, max_pairs_per_cluster=200)

    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}
    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


def arrange_choices_penalty(pure_choices_penalties, pure_choices_indexes):
    results = defaultdict(dict)
    for penalty, choice_index in zip(pure_choices_penalties, pure_choices_indexes):
        q_id, c_id = choice_index.split("_")
        q_id, c_id = int(q_id), c_id.strip()
        results[q_id][c_id] = penalty
    return results


src_path = "/data/hanxiao36/projects/RayRewardServer/questions_results_with_scores_and_penalty_qwen_question.json"
dst_path = "/data/hanxiao36/projects/RayRewardServer/questions_results_with_scores_and_penalty_qwen_question_and_choices.json"

embedder = Embedder("/data/share/250911/models/Qwen3-Embedding-0.6B")

with open(src_path, "r", encoding="utf-8") as f:
    data = json.load(f)

all_choices = []
all_choices_indexes = []

for q_id, item in enumerate(data):
    choices = item["qa"].get("choices", None)
    if not choices:
        continue
    for c_id, choice in choices.items():
        choice_idx = f"{q_id}_{c_id}"
        all_choices.append(choice)
        all_choices_indexes.append(choice_idx)
print(len(all_choices))
# diagnostics=True 会打印阈值合理性统计
penalty = cluster_share_per_problem(all_choices, distance_threshold=0.15, linkage="average", diagnostics=True)
penalty = arrange_choices_penalty(penalty, all_choices_indexes)
# print(penalty)

for i, item in enumerate(data):
    item["qwen_choices_penalty"] = penalty.get(i, None)

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
