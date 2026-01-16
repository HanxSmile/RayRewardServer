from .base import BaseHandler
import torch
from commons.registry import registry
from vllm import LLM


@registry.register_handler("qwen_penalty")
class QwenPenaltyHandler(BaseHandler):
    TASK_QUERY = {
        "single_choice": "Given a single-choice question query, retrieve another one whose content is most similar to the query",
        "multi_choice": "Given a multiple-choice question query, retrieve another one whose content is most similar to the query",
        "closed_ended": "Given a question as query, retrieve another one whose content is most similar to the query",
        "open_ended": "Given a question as query, retrieve another one whose content is most similar to the query",

        "pure_question": "Given a question as query, retrieve another one whose content is most similar to the query",
        "pure_choices": "Given a medical-related phrase as query, retrieve another one whose content is most similar to the query",
    }

    def __init__(self, model_path, batch_size=128):
        self.llm = LLM(
            model=model_path,
            task="embed"
        )
        self.batch_size = batch_size

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

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

    @staticmethod
    def format_question(question, choices=None):
        if not question:  # 虽然不valid，但是还是参与计算，防止输入输出长度不一致
            return ''
        if choices is None:  # 没有选择题，则直接返回问题
            return question
        options = []
        for idx, option in choices.items():
            options.append(f"{idx}: {option}")
        options = sorted(options, key=lambda x: x.split(":")[0])
        choices_str = ", ".join(options)
        return f"Question: {question}\n Choices: [{choices_str}]"

    def process_batch(self, batch):
        task_queries = []
        texts = []
        for item in batch:
            question = item["question"]
            choices = item["choices"]
            question_type = item["question_type"]
            text = self.format_question(question, choices)
            texts.append(text)
            task_query = self.TASK_QUERY[question_type]
            task_queries.append(task_query)
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
        batch = [{"question": "test", "choices": None, "question_type": "open_ended"} for _ in range(self.batch_size)]

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
