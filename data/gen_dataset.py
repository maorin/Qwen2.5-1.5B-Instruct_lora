"""Generate synthetic SFT samples for cloud-platform tool calling.

每条样本是一段完整的对话 (messages + tools)，会在 train 阶段被
tokenizer.apply_chat_template 渲染成 Qwen2.5 原生的 tool-calling 格式。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.cloud_tools import (  # noqa: E402
    FLAVORS,
    IMAGES,
    REGIONS,
    TOOLS,
    VM_STATUSES,
    dispatch,
    reset_state,
)

SYSTEM_PROMPT = (
    "你是一名云平台运维助手。根据用户请求，"
    "在需要调用云平台 API 时输出 <tool_call> 标签包裹的 JSON；"
    "当用户只是闲聊或问题与云平台无关时，直接用中文回答，不要调用工具。"
)


# ---- helpers ---------------------------------------------------------------
def _rand_name(prefix: str = "vm") -> str:
    return f"{prefix}-{random.choice(['web', 'db', 'cache', 'job', 'api'])}-{random.randint(1, 99)}"


def _ensure_vm() -> str:
    """Make sure at least one VM exists in state and return one id."""
    from tools.cloud_tools import _VMS  # noqa: PLC0415
    if not _VMS:
        dispatch("create_vm", {
            "name": _rand_name(), "region": random.choice(REGIONS),
            "image": random.choice(IMAGES), "flavor": random.choice(FLAVORS),
        })
    return random.choice(list(_VMS.keys()))


# ---- per-tool sample builders ---------------------------------------------
def sample_create_vm() -> dict:
    name = _rand_name()
    region = random.choice(REGIONS)
    image = random.choice(IMAGES)
    flavor = random.choice(FLAVORS)
    count = random.choices([1, 2, 3, 5], weights=[6, 2, 1, 1])[0]
    templates = [
        f"在 {region} 帮我开 {count} 台 {flavor} 的 {image} 虚拟机，命名 {name}",
        f"create {count} {image} VM(s) in {region}, name={name}, flavor={flavor}",
        f"我要在 {region} 区开一台名为 {name} 的 {image} 机器，规格 {flavor}" if count == 1 else
        f"在 {region} 给我整 {count} 台 {flavor} 的 {image}，叫 {name}",
        f"region={region} image={image} flavor={flavor} name={name} count={count}，开机",
    ]
    args = {"name": name, "region": region, "image": image, "flavor": flavor}
    if count > 1:
        args["count"] = count
    return _one_call_sample(random.choice(templates), "create_vm", args)


def sample_list_vms() -> dict:
    region = random.choice(REGIONS + [None, None])
    status = random.choice(VM_STATUSES + [None, None])
    args = {k: v for k, v in {"region": region, "status": status}.items() if v}
    parts = []
    if status:
        parts.append({"running": "运行中", "stopped": "已停止", "pending": "正在启动", "error": "异常"}[status])
    if region:
        parts.append(f"{region} 区")
    desc = "".join(parts) if parts else "所有"
    templates = [
        f"列出{desc}的虚拟机",
        f"查一下{desc}虚拟机有哪些",
        f"list vms" + (f" in {region}" if region else "") + (f" with status={status}" if status else ""),
        f"{desc}机器都列给我看看",
    ]
    return _one_call_sample(random.choice(templates), "list_vms", args)


def sample_get_vm() -> dict:
    vm_id = _ensure_vm()
    templates = [
        f"看下 {vm_id} 的详细信息",
        f"describe {vm_id}",
        f"{vm_id} 的配置是啥",
        f"帮我查一下虚机 {vm_id} 的状态和配置",
    ]
    return _one_call_sample(random.choice(templates), "get_vm", {"vm_id": vm_id})


def sample_stop_vm() -> dict:
    vm_id = _ensure_vm()
    templates = [
        f"把 {vm_id} 停掉",
        f"shutdown {vm_id}",
        f"关机 {vm_id}",
        f"我现在不用 {vm_id} 了，先停了",
    ]
    return _one_call_sample(random.choice(templates), "stop_vm", {"vm_id": vm_id})


def sample_start_vm() -> dict:
    vm_id = _ensure_vm()
    templates = [
        f"启动 {vm_id}",
        f"start {vm_id} please",
        f"把 {vm_id} 拉起来",
        f"开机 {vm_id}",
    ]
    return _one_call_sample(random.choice(templates), "start_vm", {"vm_id": vm_id})


def sample_delete_vm() -> dict:
    vm_id = _ensure_vm()
    templates = [
        f"删除 {vm_id}",
        f"销毁虚机 {vm_id}",
        f"terminate {vm_id}",
        f"{vm_id} 用不到了，删了吧",
    ]
    return _one_call_sample(random.choice(templates), "delete_vm", {"vm_id": vm_id})


def sample_get_metrics() -> dict:
    vm_id = _ensure_vm()
    metric = random.choice(["cpu", "memory", "disk_io"])
    window = random.choice([5, 10, 15, 30, 60, 120])
    label = {"cpu": "CPU", "memory": "内存", "disk_io": "磁盘 IO"}[metric]
    templates = [
        f"看下 {vm_id} 最近 {window} 分钟的{label}",
        f"{vm_id} 这一会儿{label}怎么样，最近 {window} 分钟",
        f"metrics {metric} {vm_id} last {window}m",
        f"帮我查 {vm_id} 的{label}使用情况，时间窗口 {window} 分钟",
    ]
    return _one_call_sample(
        random.choice(templates), "get_metrics",
        {"vm_id": vm_id, "metric": metric, "window_minutes": window},
    )


# ---- negative samples (don't call any tool) -------------------------------
NEGATIVE_PAIRS = [
    ("你是谁？", "我是一名云平台运维助手，可以帮你管理虚拟机、查指标、做日常运维操作。"),
    ("hi", "你好，需要我帮你做什么云平台操作？"),
    ("Kubernetes 和 Docker 啥区别？",
     "Docker 是容器运行时，负责把应用打包并运行单个容器；Kubernetes 是容器编排平台，负责跨多节点调度、扩缩容、服务发现等。"),
    ("帮我写个冒泡排序",
     "这个问题与云平台无关，我专注于云资源管理；如果你想在某台虚机里跑代码，可以告诉我虚机 id。"),
    ("今天天气怎么样？", "我没法查天气，我只能做云平台相关的操作。"),
]


def sample_negative() -> dict:
    user, assistant = random.choice(NEGATIVE_PAIRS)
    return {
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


# ---- two-turn samples (call tool, observe result, summarize) --------------
def sample_two_turn_list_then_summary() -> dict:
    # seed some VMs so list returns something non-trivial
    for _ in range(random.randint(1, 4)):
        dispatch("create_vm", {
            "name": _rand_name(), "region": random.choice(REGIONS),
            "image": random.choice(IMAGES), "flavor": random.choice(FLAVORS),
        })
    region = random.choice(REGIONS + [None])
    args = {"region": region} if region else {}
    user = (f"看下 {region} 区当前有哪些虚机" if region else "现在云上一共有多少虚机？")
    result = dispatch("list_vms", args)
    n = result.get("count", 0)
    summary = (
        f"{region} 区共有 {n} 台虚机。" if region else
        f"当前一共有 {n} 台虚机。"
    ) + ("" if n == 0 else " 详情见上方返回。")
    return {
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "list_vms", "arguments": args}}],
            },
            {"role": "tool", "name": "list_vms", "content": json.dumps(result, ensure_ascii=False)},
            {"role": "assistant", "content": summary},
        ],
    }


# ---- shared one-shot tool-call sample shape -------------------------------
def _one_call_sample(user: str, tool_name: str, arguments: dict) -> dict:
    return {
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": tool_name, "arguments": arguments}}
                ],
            },
        ],
    }


GENERATORS = [
    sample_create_vm,
    sample_list_vms,
    sample_get_vm,
    sample_stop_vm,
    sample_start_vm,
    sample_delete_vm,
    sample_get_metrics,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg_ratio", type=float, default=0.1, help="比例: 不调工具的样本")
    ap.add_argument("--multi_ratio", type=float, default=0.15, help="比例: 多轮(调用+总结)样本")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_neg = int(args.n * args.neg_ratio)
    n_multi = int(args.n * args.multi_ratio)
    n_single = args.n - n_neg - n_multi

    samples: list[dict] = []
    for _ in range(n_single):
        reset_state()
        samples.append(random.choice(GENERATORS)())
    for _ in range(n_multi):
        reset_state()
        samples.append(sample_two_turn_list_then_summary())
    for _ in range(n_neg):
        samples.append(sample_negative())

    random.shuffle(samples)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"wrote {len(samples)} samples → {out_path}")
    print(f"  single-call: {n_single}  multi-turn: {n_multi}  negative: {n_neg}")


if __name__ == "__main__":
    main()
