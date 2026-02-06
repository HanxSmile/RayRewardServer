from .base import BaseHandler
from commons.registry import registry
from .questioner_utils.vllm_utils import build_prompt_and_images, process_single

from vllm import LLM, SamplingParams
from transformers import AutoProcessor


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

    def process_batch(self, batch):
        system_prompts = [item["system"] for item in batch]
        choices = [item.get("choices", None) for item in batch]
        questions = [item['question'] for item in batch]
        types = [item['question_type'] for item in batch]
        image_list = [item['image'] for item in batch]

        valid_chats = [build_prompt_and_images(imgs, sys_p, p, c, t, self.processor) for imgs, sys_p, p, c, t in
                       zip(image_list, system_prompts, questions, choices, types)]
        responses = self.llm.generate(valid_chats, sampling_params=self.sampling_params, use_tqdm=True)

        results_all = []
        response_idx = 0
        for q, t in zip(questions, types):
            try:
                response = responses[response_idx]
                response_idx += 1
                item = process_single(q, t, response)
                results_all.append(item)

            except Exception as e:
                # Catch any other unexpected exceptions from within process_single.
                print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')
                print(f'[server] Error details: {e}')
                results_all.append({
                    'question': q,
                    'type': t,
                    'score': -1,
                    'results': [],
                    'error': f'unhandled exception in process_single: {str(e)}'
                })
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
