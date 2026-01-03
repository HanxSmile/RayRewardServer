# handlers/dummy.py
from __future__ import annotations

from typing import Any, List
from .base import BaseHandler
from commons.registry import registry


@registry.register_handler("dummy")
class DummyHandler(BaseHandler):
    """示例 handler：
    - 输入如果是数字，输出 x * scale
    - 输入如果是 {"value": x} dict，修改 value
    - 其他类型原样返回
    """

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)
        self._seen_requests = 0

    def __call__(self, item: Any) -> Any:
        self._seen_requests += 1

        if isinstance(item, (int, float)):
            return item * self.scale

        if isinstance(item, dict) and "value" in item:
            out = dict(item)
            out["value"] = out["value"] * self.scale
            return out

        return item

    def process_batch(self, items: List[Any]) -> List[Any]:
        return [self.__call__(x) for x in items]

    def health(self):
        """自定义健康检查示例。

        真正业务里：
        - 可以检查 GPU 是否 OOM 过
        - 模型是否载入完成
        - 最近错误率等
        """
        return {
            "ok": True,
            "scale": self.scale,
            "seen_requests": self._seen_requests,
        }
