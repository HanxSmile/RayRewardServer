import json
import time
import jieba
import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering


def judge_language_zh_or_eng(text):
    """
    判断字符串中中文字符的占比
    如果中文字符占比超过10%返回"zh"，否则返回"eng"
    """
    if not text:
        return "en"

    chinese_count = 0
    total_count = len(text)

    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1

    chinese_ratio = chinese_count / total_count

    if chinese_ratio > 0.1:
        return "zh"
    else:
        return "en"


def tokenize_sent(sentence):
    lang = judge_language_zh_or_eng(sentence)
    if lang == "en":
        results = sentence.split()
    else:  # zh
        results = list(jieba.cut(sentence))

    results = [_.strip() for _ in results if _.strip()]
    return results


def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [tokenize_sent(sentences[j])]
                hyp = tokenize_sent(sentences[i])
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
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
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    _print_threshold_diagnostics(dist_mat, labels, top_k=10, sample_top_clusters=5, max_pairs_per_cluster=200)
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



src_path = "/data/hanxiao36/projects/RayRewardServer/questions_results_with_scores_and_penalty_qwen.json"
dst_path = "/data/hanxiao36/projects/RayRewardServer/questions_results_with_scores_and_new_penalty.json"

with open(src_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# questions = [_["qa"]["question"] for _ in data]
questions = [format_question(_["qa"]["question"], _["qa"]["choices"]) for _ in data]

penalty = cluster_share_per_problem(questions)

results = []
for i, item in enumerate(data):
    item["new_penalty"] = penalty[i]
    results.append(item)

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(results, f)
