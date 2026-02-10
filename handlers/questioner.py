from .base import BaseHandler
from commons.registry import registry
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image


@registry.register_handler("questioner")
class QuestionerHandler(BaseHandler):
    def __init__(self, model_path, tensor_parallel_size, max_model_len, gpu_mem_util,
                 max_tokens, temperature, top_p, num_samples):
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

    def build_prompt_and_images(
            self,
            image_list,
            system_prompt,
            prompt,
    ):
        if isinstance(image_list, str):
            image_list = [image_list]
        image_content = []
        for image_path in image_list:
            image_content.append({
                "type": "image",
                "image": Image.open(image_path).convert('RGB'),
                "max_pixels": 1024 ** 2,
                "min_pixels": 512 ** 2,
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        results = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs}
        }
        return results

    def process_batch(self, batch):
        system_prompts = [item["system_prompt"] for item in batch]
        prompts = [item['prompt'] for item in batch]
        image_list = [item['images'] for item in batch]

        valid_chats = [self.build_prompt_and_images(imgs, sys_p, p) for imgs, sys_p, p in
                       zip(image_list, system_prompts, prompts)]
        responses = self.llm.generate(valid_chats, sampling_params=self.sampling_params, use_tqdm=True)
        results = []
        for i, item in enumerate(batch):
            response_list = responses[i]
            raw_response_list = [out.text for out in response_list.outputs]
            item["responses"] = raw_response_list
            results.append(item)
        return results

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
