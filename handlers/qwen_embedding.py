from .base import BaseHandler
import torch
from commons.registry import registry
from vllm import LLM


@registry.register_handler("qwen_embedding")
class QwenEmbeddingHandler(BaseHandler):

    def __init__(self, model_path, batch_size=128):
        self.llm = LLM(
            model=model_path,
            task="embed"
        )
        self.batch_size = batch_size

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        if task_description:
            return f'Instruct: {task_description}\nQuery:{query}'
        else:
            return query

    def compute_embeddings(self, texts, task_querys=None, normalize=True):
        if task_querys:
            texts = [self.get_detailed_instruct(task_query, t) for task_query, t in zip(task_querys, texts)]

        outputs = self.llm.embed(texts)
        emb = torch.tensor([o.outputs.embedding for o in outputs], dtype=torch.float32)

        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        emb = emb.cpu().detach().numpy()
        results = []
        for vec in emb:
            vec = vec.tolist()
            vec = [float(_) for _ in vec]
            results.append(vec)

        return results

    def process_batch(self, batch):
        task_queries = []
        texts = []
        for item in batch:
            prompt = item["prompt"]
            instruction = item["instruction"]
            texts.append(prompt)
            task_queries.append(instruction)

        texts_list = self.chunk_list(texts, self.batch_size)
        task_queries_list = self.chunk_list(task_queries, self.batch_size)

        total_results = []
        for t_lst, q_lst in zip(texts_list, task_queries_list):
            embeddings = self.compute_embeddings(t_lst, q_lst)
            total_results.extend(embeddings)
        return total_results

    @staticmethod
    def chunk_list(lst, chunk_size):
        """
        将列表按照指定大小分割成多个子列表

        参数:
        lst: 要分割的原始列表
        chunk_size: 每个子列表的大小

        返回:
        分割后的子列表列表
        """
        if not isinstance(lst, list):
            raise TypeError("第一个参数必须是列表")

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size必须是正整数")

        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def health(self):
        batch = [{"prompt": "test", "instruction": None} for _ in range(self.batch_size)]

        try:
            self.process_batch(batch)
            ok = True
            error = None
        except Exception as e:
            ok = False
            error = f"{type(e).__name__}: {e}"

        return {
            "ok": ok,
            "error": error,
        }
