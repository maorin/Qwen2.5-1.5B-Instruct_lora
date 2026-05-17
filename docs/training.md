# 训练流程

LoRA 微调 Qwen2.5-1.5B-Instruct 让它稳定生成云平台 function call 的完整管线。
两个核心文件：

- `data/gen_dataset.py` — 合成 SFT 训练样本
- `train/train_lora.py` — LoRA SFT 训练脚本

## 一、整体流程（6 阶段）

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. 合成数据                                                              │
│    data/gen_dataset.py 按比例生成 4 类样本                               │
│    → data/train.jsonl  (每行一条 {"tools":..., "messages":[...]})        │
├──────────────────────────────────────────────────────────────────────────┤
│ 2. 加载                                                                  │
│    pick_device() → cuda(bf16) / mps(fp16) / cpu(fp32)                    │
│    AutoTokenizer + AutoModelForCausalLM (Qwen/Qwen2.5-1.5B-Instruct)     │
│    model.config.use_cache = False  (gradient checkpointing 必需)         │
├──────────────────────────────────────────────────────────────────────────┤
│ 3. Tokenize + mask                                                       │
│    对每条样本:                                                           │
│      full_text   = apply_chat_template(messages, tools=tools)            │
│      prompt_text = apply_chat_template(messages[:-1], add_gen_prompt=T)  │
│      labels      = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]   │
│    → 模型只在"最后一条 assistant 消息"上算 loss                          │
├──────────────────────────────────────────────────────────────────────────┤
│ 4. LoRA 配置                                                             │
│    target_modules = q/k/v/o + gate/up/down proj  (全部 7 个线性层)       │
│    r=16, alpha=32, dropout=0.05                                          │
│    可训练参数 ≈ base model 的 1.5%, 显存 ~6GB                            │
├──────────────────────────────────────────────────────────────────────────┤
│ 5. SFTTrainer 训练                                                       │
│    epochs=3, lr=2e-4, cosine schedule, warmup 3%                         │
│    batch_size=2 × grad_accum=8 → 等效 16                                 │
│    gradient_checkpointing=True (省显存, 换约 30% 速度)                   │
│    DataCollatorForSeq2Seq padding (labels 用 -100 pad)                   │
├──────────────────────────────────────────────────────────────────────────┤
│ 6. 保存                                                                  │
│    trainer.save_model(out_dir)   ← 只存 LoRA adapter (~30MB), 非全权重   │
│    tokenizer.save_pretrained(out_dir)                                    │
│    可选 quick_demo 对比 fine-tune 前后输出                               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 二、数据集结构

`gen_dataset.py` 按 4 类样本拼出训练集，每类有明确教学意图：

| 类别 | 默认占比 | 形态 | 学习目标 |
|---|---|---|---|
| **single-call** | 60% | 1 个用户请求 → 1 个 `tool_call` | 7 个工具各自的参数边界 |
| **multi-call** | 20% | 1 个用户请求 → **多个** `tool_call`（一条 assistant 消息里）| 并行规划：要开两台不同名机器就发两次 `create_vm` |
| **multi-turn** | 12% | 调用 → tool 返回 → assistant 总结 | 读懂 tool 结果，如实汇报 |
| **negative** | 8% | 用户闲聊 / 问题与云无关 → 纯文本回复 | 不要 over-trigger tool call |

样本格式（一行 jsonl）：

```json
{
  "tools": [{"type":"function","function":{"name":"create_vm", ...}}, ...],
  "messages": [
    {"role":"system","content":"你是云平台运维助手..."},
    {"role":"user","content":"在 cn-east-1 帮我开一台 4c8g 的 ubuntu"},
    {"role":"assistant","content":"",
     "tool_calls":[{"type":"function","function":{"name":"create_vm","arguments":{...}}}]}
  ]
}
```

工具 schema 直接 import 自 `tools/cloud_tools.TOOLS`，**单一事实源** —
schema 改一处，训练/推理/MCP server 三处同步。

---

## 三、Tokenize + mask 策略

新版 TRL (1.x) 删了 `DataCollatorForCompletionOnlyLM`，自己手写最稳：

```python
full_text   = tokenizer.apply_chat_template(messages, tools=tools, ...)
prompt_text = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, ...)

full_ids   = tokenizer(full_text,   add_special_tokens=False).input_ids
prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

prompt_len = min(len(prompt_ids), len(full_ids))   # 安全裁剪
labels = [-100] * prompt_len + full_ids[prompt_len:]
```

效果：

```
input_ids:  <system + 7 个 tool schema + user msg + <|im_start|>assistant\n   + <tool_call>...</tool_call><|im_end|>
labels:     -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100  + <tool_call>...</tool_call><|im_end|>
            └───────────────── 不算 loss ─────────────────┘                    └──── 只在这段算 loss ────┘
```

- 验证过的比例：典型样本 153 tokens 里 135 被 mask（prompt），18 参与 loss（最后的 assistant turn）
- 多轮样本（multi-turn）只在**最后一条 assistant 总结**上算 loss；中间的 tool_call
  虽然不算，但 single-call 类样本里同样的 tool_call 模式覆盖很多次，模型仍学得到

---

## 四、LoRA 配置选择

```python
LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
```

为什么这么选：

| 参数 | 选择 | 理由 |
|---|---|---|
| `r=16` | 中等 | r=8 容量不够，r=32 在 1.5B 模型上过拟合风险 |
| `lora_alpha=32` | `2×r` | LoRA 论文常见 ratio，等价 lr scale ×2 |
| `target_modules` | 全部 7 个线性 | 只 q/k/v 学不动 function-call 这种结构化输出；带上 MLP 三层效果显著 |
| `dropout=0.05` | 轻微 | 防止 800–1200 这种小数据集过拟合 |
| `bias="none"` | 不学 bias | 节省参数，无明显收益丢失 |

可训练参数约占 base model 1.5%（~20M / ~1.5B），最终 adapter 权重 **~70MB**（bf16/fp16
存储），相比 base model 的 2.9GB 压缩到约 2.4%。

---

## 五、TRL 1.x / transformers 5.x 适配点

环境装的是 `trl 1.4.0` + `transformers 5.8.1`，API 跟早期教程差别大，关键点：

| 旧 API（TRL 0.11 时代教程）| 新 API（TRL 1.x） |
|---|---|
| `from trl import DataCollatorForCompletionOnlyLM` | **删除** — 自己预 tokenize 做 mask |
| `SFTTrainer(..., tokenizer=tok)` | `SFTTrainer(..., processing_class=tok)` |
| `SFTConfig(max_seq_length=...)` | `SFTConfig(max_length=...)` |
| 先 `get_peft_model(model, cfg)` 再传 | `SFTTrainer(..., peft_config=cfg)` 直接传 |
| `formatting_func=fn` 让 SFTTrainer 渲染 | 自己预 tokenize + `dataset_kwargs={"skip_prepare_dataset": True}` |

如果你看到老教程用 `tokenizer=`、`max_seq_length=`、`DataCollatorForCompletionOnlyLM`，
直接按上面对照表替换，否则会 import 失败或参数报错。

---

## 六、硬件自适应

```python
def pick_device():
    if torch.cuda.is_available(): return "cuda", torch.bfloat16
    if torch.backends.mps.is_available(): return "mps", torch.float16
    return "cpu", torch.float32
```

实测耗时（1200 样本 × 3 epoch）：

| 设备 | dtype | 单 epoch | 备注 |
|---|---|---|---|
| RTX 4090 | bf16 | ~2 min | grad_accum 可降到 4 |
| M-series Mac (MPS) | fp16 | ~7 min | 默认 batch_size=2 |
| CPU | fp32 | ~小时级 | 不建议，仅冒烟用 |

MPS 用 fp16 是因为它对 bf16 支持不稳；CUDA 优先 bf16 因为数值范围更大。

---

## 七、运行命令

```bash
# 1. 生成 1200 条训练数据
python data/gen_dataset.py --out data/train.jsonl --n 1200

# 2. LoRA 微调，保存到 v2 目录 (保留旧 ckpt 方便对比)
python train/train_lora.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --data data/train.jsonl \
    --out_dir checkpoints/qwen-cloud-lora-v2 \
    --epochs 3

# 常用调参
#   --lr 2e-4         默认, 太大会过拟合
#   --lora_r 16       默认, 改 32 加容量
#   --batch_size 2    M 系列 Mac 建议保持 2; CUDA 可以加到 4-8
#   --grad_accum 8    与 batch_size 配合, 等效 batch=16
#   --max_len 2048    样本一般 <500 tokens, 留充足头部
#   --no_demo         跳过 fine-tune 前后对比生成 (节省 1 分钟)
```

跑完产物（**527MB**，但推理部署只需要 **81MB**）：

```
checkpoints/qwen-cloud-lora-v2/
├── adapter_config.json          # LoRA 配置 (1KB)
├── adapter_model.safetensors    # LoRA 权重 (70MB)            ← 推理只要这个
├── chat_template.jinja          # Qwen 模板拷贝 (2KB)         ← 和这个
├── tokenizer.json               # 分词器 (11MB)               ← 和这些
├── tokenizer_config.json        # (1KB)
├── training_args.bin            # 训练参数快照 (6KB)
├── README.md                    # PEFT 自动生成
├── checkpoint-150/              # 第 2 epoch 末完整训练状态 (223MB)  ← 仅 resume 训练用
└── checkpoint-225/              # 第 3 epoch 末完整训练状态 (223MB)  ← 仅 resume 训练用
```

`checkpoint-N` 目录里除了 LoRA 权重，还存了 AdamW optimizer state (一阶+二阶矩，
比权重还大)、RNG state、lr scheduler 进度，用于"中断后接着训"。**训练完成后可以
安全删掉**：

```bash
# 训完只保留可推理产物, 每个 ckpt 目录省 ~446MB
rm -rf checkpoints/qwen-cloud-lora-v2/checkpoint-*
```

如果想让 train_lora.py 默认只留 1 个中间 ckpt（而不是 2 个），把 `SFTConfig` 里的
`save_total_limit=2` 改成 1。

加载方式（已在 `infer/agent_loop.py` 中实现）：

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "checkpoints/qwen-cloud-lora-v2")
```

---

## 八、Case study：v1 → v2 的演进

| 版本 | 数据量 | 改动 | 解决的问题 |
|---|---|---|---|
| **v1** | 800 | 只有 single-call + multi-turn + negative | baseline |
| **v2** | 1200 | 加入 240 条 multi-call 样本 | 并行规划：开两台不同名机器要发两次调用 |

同一条 query 跑两版的对比详见 [agent_loop.md §六](agent_loop.md)。结论：

> POC 量级里，schema bleeding 和并行规划这类"小模型典型短板"，**不需要换大模型 ——
> 240 条针对性样本就能在 LoRA 上推平**。前提是数据里要明确呈现"做对的样子"，
> 比单纯堆样本量更高效。

---

## 九、可能踩的坑

| 现象 | 根因 | 修法 |
|---|---|---|
| `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM'` | TRL 1.x 删除了它 | 见本文第五节 |
| `'list' object has no attribute 'keys'` 在 tokenizer init | 跑到了 base env 的老 transformers | `conda activate qwen-cloud-poc` |
| `import torch` 自身崩在 `_ctypes` | `.zshrc` 全局 export 了别的 env 的 `DYLD_LIBRARY_PATH` | 检查并删除 shell 启动文件里的相关 export |
| 训练 loss 一直在 6-8 不降 | mask 出 bug，全样本都 -100 了 | 打印一条 sample 的 labels，看有没有非 -100 段 |
| 训完模型行为没变 | 加载推理时忘了 `PeftModel.from_pretrained` 叠 adapter | 检查 `infer/agent_loop.py:43-44` 那段 |
