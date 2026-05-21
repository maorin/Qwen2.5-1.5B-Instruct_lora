# HCI 工具集成：两个 endpoint + production-style facade

POC 的"工具层"对接的是用户真实 HCI 平台 (192.168.7.x 集群)，目前两个端点：

| 工具名 (LLM 看到的) | 真实 HCI endpoint | 用途 |
|---|---|---|
| `create_vm` | `POST /v5/vm/create` | **自定义规格**建机器 (用户报 CPU/内存/磁盘) |
| `create_vm_from_template` | `POST /v5/vm/create/use/template` | **从模板**克隆 (规格继承自模板) |

---

## 一、整体架构：production-style facade

```
┌──────────────────────────────────────────────────────────────────────────┐
│  用户自然语言                                                            │
│     "建一台 4c8g 的 kylin，叫 web-1，系统盘 100G"                        │
│  or "用 ubuntu-22.04 模板克隆一台，叫 web-1"                             │
└──────────────────────┬───────────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  LLM (Qwen2.5-1.5B + LoRA)                                               │
│  输出 <tool_call>{name, arguments}</tool_call>                           │
│                                                                          │
│  ★ 关键：LLM 只看简化 schema (5-8 字段, 都是它能从原话推出来的)         │
│    - vm_name / vcpu / memory_gb / disk_gb / os_type / remark ...         │
│    - 不写 UUID, 不写 boilerplate                                         │
└──────────────────────┬───────────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  tools/cloud_tools.py::dispatch()                                        │
│  ─ 剔除 None 值                                                          │
│  ─ inspect handler 签名, 透传 cookies (浏览器会话)                       │
│  ─ 调对应 _create_vm / _create_vm_from_template                          │
└──────────────────────┬───────────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  _create_vm* (handler)                                                   │
│  ─ 用 DEFAULT_ENV 补 host_pid / host_id / pool_id / net_id / switch_id   │
│  ─ 调 _resolve_disk_id / _resolve_template_id 把名字→UUID                │
│  ─ 组装完整 payload (30+ 字段 / 5 字段)                                  │
│  ─ 调 _post_to_hci 真发 HTTP (HCI_DRY_RUN=1 时跳过)                      │
└──────────────────────┬───────────────────────────────────────────────────┘
                       ▼
                  HCI API (192.168.7.90:8006)
```

**为什么这么设计**：1.5B 小模型让它直接吐 UUID / 拼路径 / 重复 boilerplate
会很脆 — 经常漏字段、UUID 乱抄。Python 侧做胶水补字段，模型只学
"从用户原话提取核心语义参数"，这一段它学得很扎实。

---

## 二、两个工具的对照

### `create_vm` (自定义规格)

LLM 看到的简化 schema (8 字段)：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `vm_name` | string | ✅ | 虚机名 |
| `vcpu` | int | ✅ | CPU 核数 |
| `memory_gb` | int | ✅ | 内存 GB |
| `disk_gb` | int | ✅ | 系统盘 GB |
| `os_type` | int (1/2) | ✅ | 1=Linux 家族, 2=Windows |
| `remark` | string | – | 备注 |
| `display_protocol` | "spice"/"vnc" | – | 默认 spice |
| `power_on` | bool | – | 建完是否开机 |

dispatch 展开后的真实 payload (30+ 字段)：保留所有 LLM 输入的字段，再加
`host_ip`, `host_pid`, `virtualization`, `allowmc`, `safe_status`,
`numa_cpu`, `mem_strategy`, `numa_mem`, `input_devices`, `disk[]`
(含 `pool_id`/`disk_id`/`path`), `interface[]` (含 `net_id`/`switch_id`),
`cdrom[]`。

### `create_vm_from_template` (从模板克隆)

LLM 看到的简化 schema (4 字段)：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `vm_name` | string | ✅ | 新虚机名 |
| `template` | string | ✅ | 人类可读模板名 (ubuntu-22.04 / kylin-desktop / default …) |
| `count` | int | – | 同模板批量数，默认 1 |
| `remark` | string | – | 备注 |

dispatch 展开后的真实 payload (5 字段)：`template_id` (UUID 由 dispatch 解析),
`host_id` (注意是 host_id, **不是 host_pid**), `vm_name`, `count`, `remark`。

### 易踩坑：两个端点的 host UUID 是不一样的

| `/v5/vm/create` | `/v5/vm/create/use/template` |
|---|---|
| `body.host_pid` = `b7e08766-f1f9-40f3-9887-a96d3ae3b1a8` | `body.host_id` = `06c39f1a-b4d4-45be-a8a6-cb1ea168fd91` |

是两个不同的 UUID，不能复用。`DEFAULT_ENV` 分别用 `host_pid` 和
`template_host_id` 两个 key 维护。

---

## 三、UUID 解析策略

每个需要解析的 UUID 都走 **"hardcoded map → live list lookup → fallback"**
三层策略。以 `_resolve_template_id` 为例：

```
传入: template="ubuntu-22.04"
   │
   ▼
1. DEFAULT_ENV["template_id_by_name"] 有精确 key 匹配?
   有 → 返回那个 UUID, source="hardcoded_map=ubuntu-22.04"
   没 ↓
   │
   ▼
2. GET /v5/vm/template/list (带 cookies + CSRF)
   失败 → 兜底用 hardcoded 第一项,
          source="hardcoded_fallback=default (list failed: ...)"
   成功 ↓
   │
   ▼
3. 在 list 结果里:
   3a. name 精确匹配 → source="matched_name=ubuntu-22.04"
   3b. name 子串匹配 → source="matched_partial=ubuntu-22.04-lts"
   3c. 任意第一个   → source="first_template=foo (no name match for ...)"
   │
   ▼
4. 全失败 → 随机 UUID, source="fallback_uuid (...)"
          (backend 会报清晰错, 不会误用错卷)
```

每一步都打 `source` 标签写到返回结果里，方便调试 LLM 行为时回溯到底用了哪条路径。

`_resolve_disk_id` 是同一套模式，只是 list 路径换成
`/v5/storage/pool/volume/list` 且额外过滤 `domain_count==0` 选未挂载卷。

---

## 四、HTTP / Auth 配置（env vars）

dispatch 真发 HTTP 请求时这些环境变量起作用：

| env var | 默认 | 作用 |
|---|---|---|
| `HCI_API_BASE_URL` | `http://192.168.7.90:8006` | API base URL (traefik theapi entrypoint，不带 `/theapi/` 前缀) |
| `HCI_API_TOKEN` | – | 如果后端开了 Bearer 认证 |
| `HCI_API_VERIFY_SSL` | `0` | 私网集群 traefik 自签名 cert 默认不校验 |
| `HCI_DRY_RUN` | `0` | **=1 时跳过真发请求**，只返回拼好的 body (训练数据生成、demo) |
| `HCI_API_COOKIES` | – | 完整 Cookie header 串 `username=admin; sessionid=xx` |
| `HCI_SESSION_USERNAME` + `HCI_SESSION_ID` | – | 独立 env 注入 (更可读) |
| `HCI_API_ORIGIN` / `HCI_API_REFERER` | `http://localhost:3000` | list 接口 CSRF 校验用 |

Cookies 优先级：

```
dispatch(cookies=...)              ← 浏览器会话转发, 最高优先级
    > HCI_API_COOKIES               ← 完整 cookie 字符串
    > HCI_SESSION_USERNAME + ...    ← 独立 env
```

后端 Tornado 风格的 CSRF：cookie 里 `_xsrf` 自动注入到 header
`X-Xsrftoken` 和 `X-CSRFToken`，由 `_csrf_headers()` 处理。

---

## 五、训练数据：让模型学会"工具选择"

这一波最关键的学习目标 — 模型要从用户原话**二选一**：

| 用户话术信号 | 应选工具 |
|---|---|
| "X 核 Y G 内存"、"系统盘 Z G" | `create_vm` |
| "模板"、"template"、"基于 X 克隆"、"按 X 模板建" | `create_vm_from_template` |

1200 条数据的 5 桶切分（`gen_dataset.py` 默认）：

| 类别 | 占比 | 教学目标 |
|---|---|---|
| `create_vm` 单调用 | 35% | 用规格信号词 → 选 create_vm |
| `create_vm_from_template` 单调用 | 30% | 用模板信号词 → 选 from_template |
| `create_vm` 多调用 | 12% | 批量不同名定制规格 → 多次 create_vm |
| `create_vm_from_template` 多调用 | 15% | 批量从模板 → 多次 from_template |
| negative | 8% | 闲聊不调工具 |

---

## 六、验证手段

```bash
# A. dispatch 单元 (跳过 HTTP, 看 body 拼装)
HCI_DRY_RUN=1 python -c "
from tools.cloud_tools import dispatch
print(dispatch('create_vm_from_template',
               {'vm_name':'demo','template':'ubuntu-22.04','count':2}))"

# B. MCP server probe (in-process dispatch)
HCI_DRY_RUN=1 python mcp_server/server.py --probe

# C. MCP server client-probe (端到端 JSON-RPC + dispatch)
HCI_DRY_RUN=1 python mcp_server/server.py --client-probe

# D. agent_loop (LLM 端到端，验证模型有没有选对工具)
HCI_DRY_RUN=1 python infer/agent_loop.py \
    --adapter checkpoints/qwen-cloud-lora-v4 \
    --query "用 kylin-desktop 模板克隆一台，叫 demo-1"
```

`HCI_DRY_RUN=1` 让所有验证都不真发请求到集群。

---

## 七、目前还没确认的几处假设

| 假设 | 写在哪 | 真值不同时怎么改 |
|---|---|---|
| `os_type` 1=Linux, 2=Windows | `tools/cloud_tools.py` 顶部常量 | 改 `OS_TYPE_LINUX` / `OS_TYPE_WINDOWS` |
| 模板 list 端点是 `/v5/vm/template/list` | `_resolve_template_id` 里 | 改 `_get_from_hci(...)` 那行 path |
| 所有模板克隆都落到同一个 host | `DEFAULT_ENV["template_host_id"]` | 加 host 解析步骤，或暴露 host_hint 参数给 LLM |
| 单租户硬编码 DEFAULT_ENV | `tools/cloud_tools.py` 顶部 | 改成从 config / 环境变量读，schema 和训练数据都不用动 |
