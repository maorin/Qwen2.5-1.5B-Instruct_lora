# Qwen2.5-1.5B-Instruct · Cloud Function-Call POC

POC: 用 LoRA 微调 Qwen2.5-1.5B-Instruct，使其能可靠地针对自然语言指令生成
**云平台操作的 function call**，并通过 **MCP server** 把这些工具暴露给任意 MCP
客户端（Claude Desktop / Claude Code / 自研 agent 等）。

## 工程结构

```
.
├── requirements.txt
├── tools/cloud_tools.py        # 工具 schema + 内存实现 (单一事实源)
├── data/gen_dataset.py         # 合成训练样本 (instruction → tool_call)
├── train/train_lora.py         # LoRA SFT (PEFT + TRL)，自动选 mps/cuda/cpu
├── infer/agent_loop.py         # 多轮 agent: 解析 tool_call → 执行 → 喂回
├── mcp_server/server.py        # MCP stdio server，暴露同一套工具
├── docs/training.md            # 训练流程详解 (数据/tokenize/LoRA/TRL 1.x 适配)
└── docs/agent_loop.md          # agent_loop 测试流程详解 + v1/v2 实测对比
```

`tools/cloud_tools.py` 是单一事实源 — 训练样本、推理 agent、MCP server
**全部从它派生工具 schema**，确保三处定义不会漂移。

## 快速跑通 (Mac M 系列 / 单卡 GPU 均可)

```bash
# 0. 环境 — 用 conda 建虚拟环境
conda env create -f environment.yml
conda activate qwen-cloud-poc
# 之后如果只改 pip 依赖，也可以直接:
#   pip install -r requirements.txt

# 1. 生成 800 条合成训练数据
python data/gen_dataset.py --out data/train.jsonl --n 800

# 2. LoRA 微调 (Mac MPS ~20 分钟 / 4090 ~5 分钟)
python train/train_lora.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --data data/train.jsonl \
    --out_dir checkpoints/qwen-cloud-lora \
    --epochs 3

# 3. 跑 agent loop
python infer/agent_loop.py \
    --adapter checkpoints/qwen-cloud-lora \
    --query "帮我在华东2区开两台 4c8g 的 ubuntu 虚拟机，叫 web-1 和 web-2"

# 4. 启动 MCP server (stdio)，让任意 MCP 客户端调用
python mcp_server/server.py
```

## 设计要点

1. **直接复用 Qwen2.5 原生 tool-calling chat template**
   `tokenizer.apply_chat_template(msgs, tools=TOOLS, ...)` 会自动注入 `<tools>...
   </tools>` 系统段并约定 `<tool_call>{...}</tool_call>` 输出格式 — 我们只需
   让模型学会"在云平台场景下稳定按这套格式输出"。

2. **LoRA 只学输出**
   训练时把 prompt 部分的 loss mask 掉，只对 assistant 段（含 `<tool_call>`）
   计算 loss。`trl.DataCollatorForCompletionOnlyLM` 处理。

3. **合成数据多样化**
   `gen_dataset.py` 对每个工具做：参数随机抽样 + 多种中/英文表达模板 +
   单步/多步/无需调用 三类负样本，避免模型 over-trigger tool call。

4. **MCP 与微调解耦**
   MCP server 跟模型没有耦合 — 它只是把同一套工具用标准协议暴露出去。
   你既可以用微调后的 Qwen 走自己的 agent loop，也可以让 Claude / 任何 MCP 客
   户端调用同样的工具，对照效果。
