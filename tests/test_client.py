# tests/test_client.py
import requests
import yaml


def load_port_from_config(path: str = "config.yaml") -> int:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg.get("server", {}).get("port", 8000))


def test_dummy():
    port = load_port_from_config()
    url = f"http://127.0.0.1:{port}/dummy"
    payload = {"items": [1, 2, 3, 4, 5, 6, 7, 8]}
    resp = requests.post(url, json=payload)
    print("Dummy status:", resp.status_code)
    print("Dummy response:", resp.json())


def test_dummy_health():
    port = load_port_from_config()
    url = f"http://127.0.0.1:{port}/dummy/health"
    resp = requests.get(url)
    print("Dummy health status:", resp.status_code)
    print("Dummy health:", resp.json())


def test_vllm_chat():
    port = load_port_from_config()
    url = f"http://127.0.0.1:{port}/vllm_chat"
    payload = {"items": ["Hello!", "How are you?"]}
    resp = requests.post(url, json=payload)
    print("vLLM status:", resp.status_code)
    print("vLLM response:", resp.json())


if __name__ == "__main__":
    test_dummy()
    test_dummy_health()
    # 如果已经安装 vllm + 模型 OK，再打开：
    # test_vllm_chat()
