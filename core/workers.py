# workers.py
from __future__ import annotations

from commons.registry import registry
from typing import Any, Dict, List, Tuple

import ray


@ray.remote
class GPUWorker:
    """通用 GPU worker：包装一个用户自定义 Handler 类。

    - __init__ 里按 class_path 加载 handler 类并实例化；
    - infer() 负责处理一个 chunk 的 batch；
    - health() 调用 handler.health()（如果存在）并做规范化。
    """

    def __init__(self, class_name: str, handler_init_kwargs: dict | None = None):
        handler_cls = registry.get_handler_class(class_name)

        self.handler = handler_cls(**(handler_init_kwargs or {}))

    def infer(self, items_with_idx: List[Tuple[int, Any]]) -> List[Tuple[int, Any]]:
        """对一小块 (chunk) 数据做推理。

        items_with_idx: [(原始索引, item), ...]
        返回:        [(原始索引, output), ...]
        """
        if not items_with_idx:
            return []

        indices, items = zip(*items_with_idx)

        # 优先用批处理接口
        if hasattr(self.handler, "process_batch"):
            outputs = self.handler.process_batch(list(items))
        else:
            outputs = [self.handler(item) for item in items]

        # 规范化/校验 outputs，避免 zip() 静默截断导致“丢数据但不报错”
        if outputs is None:
            raise ValueError(
                f"[{type(self.handler).__name__}] process_batch/__call__ returned None; "
                f"expected a sequence with length {len(items)}."
            )

        # 允许 list/tuple 等序列；明确拒绝 str/bytes/dict（这些用 list() 会产生很诡异的结果）
        if isinstance(outputs, (str, bytes, dict)):
            raise TypeError(
                f"[{type(self.handler).__name__}] returned {type(outputs).__name__}; "
                "expected a list-like sequence of outputs."
            )

        if not isinstance(outputs, list):
            try:
                outputs = list(outputs)
            except Exception as e:
                raise TypeError(
                    f"[{type(self.handler).__name__}] returned non-iterable outputs: {type(outputs).__name__}"
                ) from e

        if len(outputs) != len(items):
            raise ValueError(
                f"[{type(self.handler).__name__}] output length mismatch: "
                f"got {len(outputs)} outputs for {len(items)} inputs. "
                "Your handler must return exactly one output per input item."
            )

        return list(zip(indices, outputs))

    def health(self) -> Dict[str, Any]:
        """真正意义上的健康检查。

        - 如果 handler 实现了 health()，就用它的结果；
        - 支持 bool / dict / 其他类型（会包一层）。
        """
        info: Dict[str, Any]

        if hasattr(self.handler, "health"):
            try:
                res = self.handler.health()
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"handler.health() raised: {type(e).__name__}: {e}",
                }

            if isinstance(res, bool):
                info = {"ok": res}
            elif isinstance(res, dict):
                # 确保有 ok 字段
                ok = res.get("ok", True)
                info = {"ok": bool(ok), **res}
            else:
                info = {"ok": True, "detail": res}
        else:
            # 没实现 health，就只能认为“看起来正常”
            info = {"ok": True, "detail": "no custom health() defined in handler"}

        return info
