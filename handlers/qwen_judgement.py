from .base import BaseHandler
import re
import xmltodict
from commons.registry import registry
from .prompts.qwen_juedgement_prompts import PROMPTS
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


@registry.register_handler("qwen_judgement")
class QwenJudgementHandler(BaseHandler):
    def __init__(
            self,
            model_path, tensor_parallel_size, max_model_len, gpu_mem_util,
            max_tokens, temperature, top_p,
            batch_size
    ):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.batch_size = batch_size

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

    @staticmethod
    def format_input(question, choices, question_type):
        if not question:
            text = ""
        else:
            options = []
            for idx, option in choices.items():
                this_option = f"{idx}: {option}"
                options.append(this_option)
            options = sorted(options, key=lambda x: x.split(":")[0])
            choices = ", ".join(options)
            choices = f"[{choices}]"
            text = f"题干: {question}\n选项: {choices}"
        prompt = PROMPTS[question_type]
        if prompt:
            prompt = prompt.format(content=text)
        else:
            prompt = text
        return prompt

    def postprocess(self, response):
        response = response.replace("```xml", "```").replace("```Xml", "```").replace("```XML", "```")

        response_list = response.split("```")
        string = ""
        for line in response_list:
            if "<response>" in line and "</response>" in line and "<result>" in line and "</result>" in line:
                string = line

        string = self.escape_label_content(string, "<result>", "</result>")
        string = self.escape_label_content(string, "<reason>", "</reason>")
        try:
            dic = xmltodict.parse(string)
            entity = dic["response"]
            result = str(entity["result"])
            reason = entity.get("reason", "Error.")
        except Exception as e:
            if ">no<" in response.lower():
                result = "no"
            else:
                result = "yes"
            reason = "Parsed Failed."
        if "no" in result.lower():
            result = False
        else:
            result = True
        item = {
            "response": response,
            "result": result,
            "reason": reason,
        }

        return item

    @staticmethod
    def escape_label_content(xml_string, tag_start, tag_end):
        def escape_content(match):
            content = match.group(1)

            escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"',
                                                                                                              '&quot;').replace(
                "'", '&apos;')
            pattern = "<text>{}</text>".replace("<text>", tag_start).replace("</text>", tag_end)
            return pattern.format(escaped_content)

        escaped_string = re.sub(r'<text>(.*?)</text>'.replace("<text>", tag_start).replace("</text>", tag_end),
                                escape_content, xml_string, flags=re.DOTALL)
        return escaped_string

    def process_batch(self, batch):

        total_model_inputs = []
        for item in batch:
            question = item["question"]
            choices = item["choices"]
            question_type = item["question_type"]
            prompt = self.format_input(question, choices, question_type)
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode='off',
            )
            total_model_inputs.append(text)

        batch_list = self.chunk_list(total_model_inputs, self.batch_size)
        total_results = []

        for micro_batch in batch_list:
            outputs = self.llm.generate(micro_batch, self.sampling_params, use_tqdm=False)
            outputs = [_.outputs[0].text for _ in outputs]
            outputs = [self.postprocess(_) for _ in outputs]
            total_results.extend(outputs)
        return total_results

    def health(self):
        batch = [{
            "question": "在评估胸部X光时，如何识别两侧无横膈剖面的情况，这是否会导致更闭合的走向？",
            "choices": {
                "A": "是",
                "B": "否",
                "C": "旁叶型断面",
                "D": "由纵隔压积和纵隔最非可逆增厚"
            },
            "question_type": "single_choice"
        } for _ in range(self.batch_size)]

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
