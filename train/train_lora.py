"""LoRA SFT on Qwen2.5-1.5B-Instruct for cloud function calling.

适配 TRL 1.x / transformers 5.x:
- 不用已删除的 DataCollatorForCompletionOnlyLM, 自己预 tokenize, 手工把 prompt
  段 labels 置 -100 (loss 只覆盖最后一条 assistant 消息);
- SFTTrainer 直接收 peft_config, 不再需要先 get_peft_model 包一层;
- 用 DataCollatorForSeq2Seq 处理变长 padding (含 labels 用 -100 pad);
- 自动选 cuda / mps / cpu 设备与对应 dtype。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from trl import SFTConfig, SFTTrainer


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


def build_tokenize_fn(tokenizer, max_length: int):
    """返回一个把 {messages, tools} 编码成 {input_ids, attention_mask, labels}
    的函数；prompt 段（除了最后一条 assistant 消息以外的全部内容）labels = -100,
    模型只在最后一条 assistant 回复上算 loss。"""

    def fn(example):
        msgs = example["messages"]
        tools = example.get("tools")

        full_text = tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True,
        )

        full_ids = tokenizer(full_text, truncation=True, max_length=max_length,
                             add_special_tokens=False).input_ids
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length,
                               add_special_tokens=False).input_ids

        # 安全裁剪: prompt_ids 不应超过 full_ids
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return fn


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

    raw = load_dataset(Path(args.data))
    tokenize_fn = build_tokenize_fn(tokenizer, args.max_len)
    tokenized = raw.map(
        tokenize_fn,
        remove_columns=raw.column_names,
        desc="tokenize+mask",
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
        save_total_limit=1,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        max_length=args.max_len,
        packing=False,
        report_to="none",
        gradient_checkpointing=True,
        # 我们已经自己做了 mask, 不让 SFTTrainer 再去渲染模板
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt",
    )

    if not args.no_demo:
        print("\n=== before fine-tune ===")
        # 用一条样本里的 tools 字段做 demo
        quick_demo(tokenizer, model, raw[0].get("tools"), device)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    if not args.no_demo:
        print("\n=== after fine-tune ===")
        quick_demo(tokenizer, trainer.model, raw[0].get("tools"), device)

    print(f"\nLoRA adapter saved → {args.out_dir}")


if __name__ == "__main__":
    main()
