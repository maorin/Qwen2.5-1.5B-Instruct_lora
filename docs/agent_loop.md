# `agent_loop.py` 测试流程与实测结果

`infer/agent_loop.py` 是本 POC 中**模型与工具的桥**：模拟"用户提问 → 模型决策 →
真实执行工具 → 把结果喂回模型 → 总结"的循环。它**不走 MCP 协议**，是一个进程
内的简化版 agent，专门用来快速验证微调后的模型行为。验证 MCP 协议层请用
`mcp_server/server.py --client-probe`。

---

## 一、整体流程（7 步）

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. 加载                                                                  │
│    pick_device() → cuda / mps / cpu                                      │
│    AutoTokenizer + AutoModelForCausalLM (base)                           │
│    PeftModel.from_pretrained(model, adapter)  ← LoRA 权重叠加上去        │
├──────────────────────────────────────────────────────────────────────────┤
│ 2. 拼初始 messages                                                       │
│    [ system: "你是云平台运维助手...",                                    │
│      user:   "帮我开两台 ubuntu 虚机..." ]                               │
├──────────────────────────────────────────────────────────────────────────┤
│ 3. 渲染 prompt（每一步重渲染）                                           │
│    tokenizer.apply_chat_template(messages,                               │
│                                  tools=TOOLS,         ← 7 个工具 schema  │
│                                  add_generation_prompt=True)             │
│    → 注入 <tools>...</tools> 系统段 + "<|im_start|>assistant\n"          │
├──────────────────────────────────────────────────────────────────────────┤
│ 4. 模型 generate                                                         │
│    do_sample=False (greedy, 可复现), max_new_tokens=256                  │
│    → 原始输出文本                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ 5. 解析输出                                                              │
│    正则 <tool_call>\s*(\{.*?\})\s*</tool_call> 抓所有 tool_call          │
│    → 0 个: 当作最终答复, 退出循环                                        │
│    → ≥1 个: 进入第 6 步                                                  │
├──────────────────────────────────────────────────────────────────────────┤
│ 6. 执行 + 回喂                                                           │
│    把这次 assistant 输出 (含 tool_calls) 追加进 messages                 │
│    对每个 tool_call:                                                     │
│      result = dispatch(name, arguments)   ← 进程内调 cloud_tools 真函数  │
│      把 {role:"tool", name, content: json(result)} 追加进 messages       │
├──────────────────────────────────────────────────────────────────────────┤
│ 7. 回到第 3 步                                                           │
│    最多 max_steps (默认 6) 轮; 通常 2 轮就结束 (调工具 → 总结)           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 二、用 `web-1` / `web-2` 走一遍

启动命令：

```bash
python infer/agent_loop.py \
    --adapter checkpoints/qwen-cloud-lora-v2 \
    --query "帮我在华东2区开两台 4c8g 的 ubuntu 虚拟机，叫 web-1 和 web-2"
```

### Round 1 (step 3–7)

`messages` = `[system, user("开两台 web-1 web-2")]`

模型 generate 后输出：

```
<tool_call>
{"name": "create_vm", "arguments": {"vm_id": null, "name": "web-1", "region": "cn-east-2", "image": "ubuntu-22.04", "flavor": "4c8g", "count": 1}}
</tool_call>
<tool_call>
{"name": "create_vm", "arguments": {"vm_id": null, "name": "web-2", "region": "cn-east-2", "image": "ubuntu-22.04", "flavor": "4c8g", "count": 1}}
</tool_call>
```

解析得到 2 个 tool_call → 不是空，执行：

| 调用 | 结果 |
|---|---|
| `dispatch("create_vm", {name:"web-1", ...})` | `{"created":[{"vm_id":"vm-1c976f30","name":"web-1",...}]}` |
| `dispatch("create_vm", {name:"web-2", ...})` | `{"created":[{"vm_id":"vm-9952ebac","name":"web-2",...}]}` |

`messages` 长度从 2 → 5（追加 1 个 assistant、2 个 tool）。

### Round 2 (step 3–7)

重渲染 `messages` → 模型看到自己之前的 tool_call 和 tool 返回的真实结果。

模型 generate 输出纯文本：

```
我成功在华东2区开了两台 4c8g 的 ubuntu 虚拟机，详情见上方返回。
你可以通过 vm-id 查看每个虚机的状态和信息。
```

解析得到 0 个 tool_call → 走 `[final assistant]` 分支，**退出循环**。

---

## 三、关键设计决策

| 设计 | 为什么这么写 |
|---|---|
| **正则解析 `<tool_call>` 标签** | 跟 Qwen2.5 原生 chat template 输出格式严格对齐；不依赖额外解析库 |
| **`do_sample=False` (greedy)** | 测试要可复现；同一个 query 每次跑结果一样 |
| **`max_steps=6`** | 防止模型陷入死循环反复调工具 |
| **进程内 `dispatch`，不走 MCP** | 测试焦点是**模型决策对不对**，剥离协议层噪声 |
| **`PeftModel.from_pretrained` 叠 adapter** | 只加载几 MB 的 LoRA delta，不重复下载 base model |

---

## 四、跟"真实 MCP 客户端"的区别

| 这里的 `agent_loop.py` | Claude Desktop / 自家产品 |
|---|---|
| 进程内函数调用 (`dispatch`) | JSON-RPC over stdio (跟 `mcp_server/server.py` 通信) |
| 一个 Python 进程 | 客户端进程 ⇄ MCP server 进程 |
| 用于**快速测模型** | 用于**实际暴露给用户** |

验证微调成果时跑 `agent_loop.py`，验证协议层时跑
`mcp_server/server.py --client-probe`。两条链路**互相独立**，可以单独修单独测。

---

## 五、能测出什么 / 测不出什么

**✅ 能测**

- 模型该不该调工具
- 调哪一个工具
- 参数填得对不对
- 对 tool 结果的总结是否如实
- 多步规划能力（一条用户请求拆成几个 tool_call）

**❌ 测不出 / 不模拟**

- MCP 协议层的兼容性
- 真实客户端拉起的稳定性
- 并发会话隔离
- 流式输出（stream）
- 用户中途打断
- tool 调用并发执行（这里是顺序 dispatch）

---

## 六、实测结果对比（v1 vs v2）

同一条 query —— `"帮我在华东2区开两台 4c8g 的 ubuntu 虚拟机，叫 web-1 和 web-2"`
—— 跑两个 LoRA 版本，结果对照如下。

| 指标 | v1 (800 样本) | v2 (1200 样本，加入 240 条 multi-call) | 状态 |
|---|---|---|---|
| **tool_call 数量** | 1 个 (`count=2`) | **2 个独立调用** | ✅ 学会了 |
| **VM 名字** | `web-1-1` / `web-1-2`（错）| `web-1` / `web-2`（对）| ✅ 修复 |
| **Schema bleeding 字段数** | 3 (`vm_id`, `metric`, `window_minutes`) | **1**（只剩 `vm_id`）| ✅ 减少 67% |
| **总结是否如实** | 编造成功（实际 tool 报 error）| 如实总结 | ✅ 不再幻觉 |

### v1 实际输出（旧 LoRA，反例）

```
<tool_call>
{"name": "create_vm", "arguments": {"vm_id": null, "metric": null, "window_minutes": null, "region": "cn-east-2", "name": "web-1", "image": "ubuntu-22.04", "flavor": "4c8g", "count": 2}}
</tool_call>
```

后端 `count=2` 触发自动加后缀，实际创建出 `web-1-1` / `web-1-2`，**名字跟用户期望的不一致**。
模型在 step 2 不顾这个事实，编出"开好了 web-1-1、web-1-2"的总结 — 双重错误。

### v2 实际输出（新 LoRA，正例）

```
<tool_call>
{"name": "create_vm", "arguments": {"vm_id": null, "name": "web-1", ..., "count": 1}}
</tool_call>
<tool_call>
{"name": "create_vm", "arguments": {"vm_id": null, "name": "web-2", ..., "count": 1}}
</tool_call>
```

两次独立 `create_vm`，名字完全匹配用户原话；schema bleeding 从 3 个 null 字段减到 1 个，
且 `dispatch` 会自动过滤 `vm_id=null`，不影响实际调用。step 2 总结如实。

---

## 七、还没根治的两个小问题

1. **`vm_id=null` 仍在 `create_vm` 里漏一个**
   - 严重度：低（`dispatch` 已经吞掉）
   - 根因：`vm_id` 出现在另外 5 个工具里（`get/start/stop/delete/get_metrics`），它的"基础概率"太高
   - 想根治：再加几十条样本 / 提高 LoRA r（现在 r=16）/ 多训 1 epoch

2. **总结没把名字念出来**
   - 这次没念"web-1 和 web-2"，只说"两台"
   - 不算 bug，只是稍微不够具体；不影响正确性

---

## 八、关键经验

> **POC 量级里，schema bleeding 和并行规划这类"小模型典型短板"，
> 不需要换大模型 —— 240 条针对性样本就能在 LoRA 上推平**。
> 前提是数据里要明确呈现"做对的样子"，比单纯堆样本量更高效。
