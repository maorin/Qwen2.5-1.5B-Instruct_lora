"""LoRA SFT on Qwen2.5-1.5B-Instruct for cloud function calling.

Key points:
- 用 tokenizer.apply_chat_template(messages, tools=tools) 渲染样本 → 完美
  契合 Qwen 原生 tool-calling 格式；
- 用 trl.DataCollatorForCompletionOnlyLM 让 loss 只覆盖 assistant 段；
- 自动选 cuda / mps / cpu 设备与对应 dtype；
- 训练前后各跑一条样例，肉眼对比 fine-tune 效果。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


# Qwen2.5 chat template 中标识 assistant 段开始的固定 token 序列。
# DataCollatorForCompletionOnlyLM 用它来定位 loss 起点。
ASSISTANT_RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_dataset(path: Path) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def make_formatter(tokenizer):
    """Return a function: example → rendered text."""

    def fmt(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tools=example.get("tools"),
            tokenize=False,
            add_generation_prompt=False,
        )

    return fmt


def quick_demo(tokenizer, model, tools, device: str) -> None:
    msgs = [
        {"role": "system", "content": "你是一名云平台运维助手。需要调用工具时输出 <tool_call> JSON。"},
        {"role": "user", "content": "在 cn-east-1 帮我开一台 4c8g 的 ubuntu-22.04，叫 web-1"},
    ]
    prompt = tokenizer.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--no_demo", action="store_true")
    args = ap.parse_args()

    device, dtype = pick_device()
    print(f"[device] {device} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    raw = load_dataset(Path(args.data))
    fmt = make_formatter(tokenizer)

    # SFTTrainer 接受一个 formatting_func 返回字符串，由它内部 tokenize。
    # 配合 DataCollatorForCompletionOnlyLM 实现 prompt 段 loss masking。
    collator = DataCollatorForCompletionOnlyLM(
        response_template=ASSISTANT_RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    sft_cfg = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        max_seq_length=args.max_len,
        packing=False,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=raw,
        formatting_func=fmt,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    if not args.no_demo:
        print("\n=== before fine-tune ===")
        quick_demo(tokenizer, model, raw[0]["tools"], device)

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    if not args.no_demo:
        print("\n=== after fine-tune ===")
        quick_demo(tokenizer, model, raw[0]["tools"], device)

    print(f"\nLoRA adapter saved → {args.out_dir}")


if __name__ == "__main__":
    main()
