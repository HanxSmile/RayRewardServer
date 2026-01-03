# handlers/vllm_chat.py
from __future__ import annotations

from typing import List, Dict, Any
from .base import BaseHandler
from commons.registry import registry

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


@registry.register_handler("vllm_chat")
class VLLMChatHandler(BaseHandler):
    """示例：把 vLLM LLM 封装成一个 batch handler。

    约定：
    - 每个 item 是一个字符串 prompt
    - 输出也是字符串
    """

    def __init__(self, model_path: str, max_tokens: int = 64, **llm_kwargs):
        if LLM is None:
            raise ImportError(
                "vllm is not installed. Please `pip install vllm` to use VLLMChatHandler."
            )

        self.model_path = model_path
        self.max_tokens = int(max_tokens)
        self._seen_requests = 0

        # 简单示例：单卡 vLLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=int(llm_kwargs.pop("tensor_parallel_size", 1)),
            **llm_kwargs,
        )
        self.sampling_params = SamplingParams(max_tokens=self.max_tokens)

    def process_batch(self, prompts: List[str]) -> List[str]:
        self._seen_requests += len(prompts)
        outputs = self.llm.generate(prompts, self.sampling_params)
        responses: List[str] = []
        for out in outputs:
            if out.outputs:
                responses.append(out.outputs[0].text)
            else:
                responses.append("")
        return responses

    def health(self) -> Dict[str, Any]:
        """简单的 vLLM 健康检查。"""
        try:
            # 非常轻量的 probe：空 prompt
            _ = self.llm.generate(["ping"], SamplingParams(max_tokens=1))
            ok = True
            error = None
        except Exception as e:
            ok = False
            error = f"{type(e).__name__}: {e}"

        return {
            "ok": ok,
            "model_path": self.model_path,
            "max_tokens": self.max_tokens,
            "seen_requests": self._seen_requests,
            "error": error,
        }
