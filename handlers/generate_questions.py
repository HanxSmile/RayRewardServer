from .base import BaseHandler
from typing import Optional, Union, Any, Dict
from io import BytesIO
import math
import re
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils import process_vision_info
from commons.registry import registry

from vllm import LLM, SamplingParams
from transformers import AutoProcessor


@registry.register_handler("generate_questions")
class GenerateQuestionsHandler(BaseHandler):
    def __init__(
            self, model_path,
            tensor_parallel_size, max_model_len, gpu_mem_util,
            max_tokens, temperature, top_p, num_samples,
            prompt_key, sys_prompt_key, image_key, image_limits=None,
            min_pixels: Optional[int] = None, max_pixels: Optional[int] = None,
            question_type_key="question_type"
    ):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            trust_remote_code=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 10},
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.0,
            n=num_samples,
        )
        self.prompt_key = prompt_key
        self.sys_prompt_key = sys_prompt_key
        self.image_key = image_key
        self.image_limits = image_limits
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.question_type_key = question_type_key

    @staticmethod
    def extract_choices(
            text: str,
            keys: str = "ABCDEF",
            ensure_all: bool = False,
    ) -> Dict[str, str]:
        """
        从文本中提取形如 A: ... B: ... C: ... D: ... 的选项内容，并转换为字典。

        依据：选项标记（如 A: B: C: D:）一定会出现。
        匹配策略：抓取 “选项标记 + 内容”，内容一直匹配到下一个选项标记或文本末尾。

        参数：
            text: 原始文本（可包含换行、标签等）
            keys: 允许的选项键，默认 "ABCD"
            ensure_all: 是否保证返回字典包含所有 keys（缺失则置空字符串）

        返回：
            dict，例如 {"A": "肺炎", "B": "肺结节", "C": "肺癌", "D": "肺气肿"}
        """
        # 动态构造 key 字符集，例如 ABCD -> [ABCD]
        key_class = re.escape(keys)

        # 核心正则：key: + 内容（懒惰）+ (到下一个key:或结尾停止)
        pattern = rf'\b([{key_class}]):\s*(.*?)(?=\s*\b[{key_class}]:|$)'
        pairs = re.findall(pattern, text, flags=re.S)

        # 清洗：去掉多余空白、以及像 [ ... ] 包裹时残留的括号
        result = {k: v.strip().strip('[],【】， \n\t') for k, v in pairs}

        if ensure_all:
            full = {k: "" for k in keys}
            full.update(result)
            return full

        return result

    def build_prompt_and_images(self, item):
        system_prompt = item[self.sys_prompt_key]
        prompt = item[self.prompt_key]
        image_lst = item[self.image_key]
        if self.image_limits is not None:
            image_lst = image_lst[:self.image_limits]
        images = [self.process_image(_, self.min_pixels, self.max_pixels) for _ in image_lst]
        image_contents = [{"type": "image", "image": images[_]} for _ in range(len(images))]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_contents + [{"type": "text", "text": prompt}]},
        ]
        input_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)
        results = {
            "prompt": input_prompt,
            "multi_modal_data": {"image": image_inputs}
        }

        return results

    @staticmethod
    def extract_question_content(response: str, question_type: str) -> str:
        assert question_type in ["multi_choice", "single_choice", "closed_ended"]
        thinking = response.split("<think>")[-1].split("</think>")[0]
        if question_type == "closed_ended":
            pattern = re.compile(r"<think>.*</think>.*<question>(.*?)</question>", re.DOTALL)
            match = re.search(pattern, response)
            try:
                return match.group(1).strip(), None, thinking
            except Exception as e:
                print(f"Error during extract question content: {e}.\n Response: {response}")
                return "", None, thinking

        else:  # single_choice or multi_choice
            pattern = re.compile(r"<think>.*</think>.*<question>(.*?)</question>.*<choices>(.*?)</choices>", re.DOTALL)
            match = re.search(pattern, response)

            if match:
                question_content = match.group(1).strip() if match.group(1) else None
                question_content = question_content or ""
                choices_content = match.group(2).strip() if match.group(2) else None
                try:
                    choices = GenerateQuestionsHandler.extract_choices(choices_content)
                except Exception as e:
                    print(f"Error during extract choices content: {e}\nResponse: {response}")
                    choices = None
                if choices:
                    return question_content, choices, thinking
                else:
                    return "", None, thinking
            return "", None, thinking

    def process_batch(self, batch):
        valid_chats = [self.build_prompt_and_images(item) for item in batch]
        responses = self.llm.generate(valid_chats, sampling_params=self.sampling_params, use_tqdm=True)

        results_all = []
        for i, (item, response) in enumerate(zip(batch, responses)):
            response = response.outputs[0].text
            try:
                question, choices, thinking = self.extract_question_content(response, item[self.question_type_key])
            except Exception as e:
                # Catch any other unexpected exceptions from within process_single.
                print(f'[server] CRITICAL: An unhandled error occurred while processing response: {response}')
                print(f'[server] Error details: {e}')
                question, choices, thinking = "", None, None
            qa = None
            if question:
                qa = {
                    "question": question,
                    "choices": choices,
                    "thinking": thinking,
                }
            item["qa"] = qa
            results_all.append(item)
        print('[server] All results have been processed.')
        return results_all

    def health(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "你好。"}]}
        ]
        # 使用 processor 的 chat template 格式化
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        dummy_inputs = {"prompt": prompt}
        try:
            _ = self.llm.generate([dummy_inputs], sampling_params=self.sampling_params, use_tqdm=True)
            ok = True
            error = None
        except Exception as e:
            ok = False
            error = f"{type(e).__name__}: {e}"

        return {
            "ok": ok,
            "error": error,
        }

    @staticmethod
    def process_image(
            image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
    ) -> ImageObject:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        image.load()  # avoid "Too many open files" errors
        if max_pixels is not None and (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if min_pixels is not None and (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
