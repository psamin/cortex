"""The planner sends at most cfg.max_images images per request (vLLM's --limit-mm-per-prompt),
newest kept, without touching the conversation it records. No simulator needed."""

import json

import httpx

from cortex.agentic.astra import AstraPlanner, PlannerConfig, Usage, window_images


def image_msg(i: int) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": f"Image {i}"},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{i}"}}]}


def urls(messages: list[dict]) -> list[str]:
    return [p["image_url"]["url"] for m in messages if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image_url"]


def test_window_keeps_the_newest_images_and_leaves_history_alone():
    history = [{"role": "system", "content": "s"}, *[image_msg(i) for i in range(10)]]
    shown = window_images(history, keep=4)
    assert urls(shown) == [f"data:image/jpeg;base64,{i}" for i in range(6, 10)]
    assert shown[1]["content"][1]["text"].startswith("[earlier image omitted")
    assert len(urls(history)) == 10  # the recorded conversation still has every image


def test_requests_never_exceed_the_image_limit():
    sent = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    planner = AstraPlanner(PlannerConfig(provider="selfhosted", api_key="none", model="Qwen/Qwen3-VL-8B-Instruct",
                                         endpoint="http://vllm.test/v1/chat/completions", price_per_m=None,
                                         max_images=4),
                           transport=httpx.MockTransport(handler))
    planner._complete([{"role": "system", "content": "s"}, *[image_msg(i) for i in range(12)]], Usage())
    assert len(urls(sent[0]["messages"])) == 4
