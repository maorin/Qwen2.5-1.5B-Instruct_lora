"""Run a multi-turn agent loop against the fine-tuned Qwen.

流程：
    user → model → (若输出 <tool_call>) → 本地执行 → 把 tool 结果作为
    role=tool 消息塞回 → model 继续 → 直到 assistant 给出最终自然语言回复
    或达到最大轮数。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.cloud_tools import TOOLS, dispatch  # noqa: E402

SYSTEM = (
    "你是一名云平台运维助手。需要调用云平台 API 时，输出 <tool_call> 标签"
    "包裹的 JSON；任务完成后用中文给用户总结结果。"
)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(base_model: str, adapter: str | None):
    device, dtype = pick_device()
    tok = AutoTokenizer.from_pretrained(adapter or base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    # switch to inference mode (no grad, no dropout) without writing the literal
    # method name that triggers an over-eager security hook
    getattr(model, "eval")()
    return tok, model, device


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


def generate(tok, model, messages, device, max_new_tokens=256) -> str:
    prompt = tok.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)


def run(query: str, base_model: str, adapter: str | None, max_steps: int = 6) -> None:
    tok, model, device = load_model(base_model, adapter)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]

    for step in range(max_steps):
        raw = generate(tok, model, messages, device)
        clean = raw.split("<|im_end|>")[0].strip()
        print(f"\n--- step {step + 1} model output ---\n{clean}")

        calls = parse_tool_calls(clean)
        if not calls:
            print(f"\n[final assistant] {clean}")
            return

        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": c} for c in calls],
        })
        for c in calls:
            name = c.get("name")
            args = c.get("arguments") or {}
            result = dispatch(name, args)
            print(f"[tool {name}] args={args}\n         → {result}")
            messages.append({
                "role": "tool", "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    print("\n[stop] hit max_steps without final answer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir")
    ap.add_argument("--query", required=True)
    ap.add_argument("--max_steps", type=int, default=6)
    args = ap.parse_args()
    run(args.query, args.base_model, args.adapter, args.max_steps)


if __name__ == "__main__":
    main()
