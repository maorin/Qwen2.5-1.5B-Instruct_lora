"""Generate synthetic SFT samples for the HCI create_vm tool.

每条样本是一段完整对话 (messages + tools), 会在 train 阶段被
tokenizer.apply_chat_template 渲染成 Qwen2.5 原生 tool-calling 格式。

样本分 3 类:
  - single-call: 用户描述一台 VM → assistant 发 1 个 create_vm
  - multi-call:  用户描述多台不同名 VM → assistant 同一条消息里发多个 create_vm
  - negative:    用户闲聊 / 与云无关 → 不调任何工具
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.cloud_tools import OS_TYPE_LINUX, OS_TYPE_WINDOWS, TOOLS  # noqa: E402

SYSTEM_PROMPT = (
    "你是一名 HCI 云平台运维助手。用户用自然语言描述想要的虚拟机时, "
    "调用 create_vm 工具来创建; 仅闲聊或与云平台无关的问题用中文直接回答, "
    "不要调用工具。环境相关字段 (host_pid / pool_id / 网络 UUID 等) 由平台 "
    "agent 自动补齐, 你只需要填用户给的核心参数。"
)


# ---- 表征用户语言意图的词典 ------------------------------------------------
LINUX_DISTROS_ZH = [
    "ubuntu", "ubuntu 22.04", "ubuntu 20.04",
    "centos", "centos 7", "centos 8",
    "debian", "debian 12", "rocky", "rocky 9",
    "kylin", "kylin desktop", "kylin server", "麒麟", "麒麟桌面",
    "uos", "统信", "openEuler", "欧拉",
    "linux",
]
WINDOWS_DISTROS_ZH = [
    "windows", "windows 10", "windows 11", "windows server",
    "win10", "win11", "win2019", "win server 2022",
]

NAME_ROOTS = ["web", "db", "cache", "api", "job", "worker", "test", "monitor", "gw"]


def _rand_vm_name() -> str:
    return f"{random.choice(NAME_ROOTS)}-{random.randint(1, 99)}"


def _rand_vcpu() -> int:
    return random.choice([1, 2, 2, 4, 4, 4, 8, 8, 16, 32])


def _rand_mem() -> int:
    return random.choice([1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 64])


def _rand_disk() -> int:
    return random.choice([20, 40, 50, 50, 100, 100, 200, 500, 1000])


def _rand_os() -> tuple[int, str]:
    """Returns (os_type_int, human_phrase_for_prompt)."""
    if random.random() < 0.7:
        return OS_TYPE_LINUX, random.choice(LINUX_DISTROS_ZH)
    return OS_TYPE_WINDOWS, random.choice(WINDOWS_DISTROS_ZH)


# ---- single-call samples ---------------------------------------------------
def sample_create_vm() -> dict:
    name = _rand_vm_name()
    vcpu = _rand_vcpu()
    mem = _rand_mem()
    disk = _rand_disk()
    os_type, distro_phrase = _rand_os()
    args: dict = {
        "vm_name": name, "vcpu": vcpu, "memory_gb": mem,
        "disk_gb": disk, "os_type": os_type,
    }

    # 30% 概率带备注
    if random.random() < 0.3:
        remark = random.choice([
            "测试用", "压测", "临时", "生产环境", "灰度环境",
            "QA 验证", "性能基线", "数据备份机",
        ])
        args["remark"] = remark
        remark_phrase = f"，备注\"{remark}\""
    else:
        remark_phrase = ""

    # 25% 概率创建后立即开机
    if random.random() < 0.25:
        args["power_on"] = True
        power_phrase = random.choice(["，建好直接开机", "，创建完启动", "并启动"])
    else:
        power_phrase = ""

    # 15% 概率指定显示协议
    if random.random() < 0.15:
        proto = random.choice(["spice", "vnc"])
        args["display_protocol"] = proto
        proto_phrase = f"，用 {proto} 显示"
    else:
        proto_phrase = ""

    templates = [
        f"帮我开一台 {vcpu}c{mem}g 的 {distro_phrase} 虚机，叫 {name}，"
        f"系统盘 {disk}G{remark_phrase}{power_phrase}{proto_phrase}",
        f"创建一台 {distro_phrase} 的虚拟机，名称 {name}，CPU {vcpu} 核，"
        f"内存 {mem}G，磁盘 {disk}G{remark_phrase}{power_phrase}{proto_phrase}",
        f"在集群上来一台 {name}：{vcpu} 核 {mem}G 内存 {disk}G 盘 {distro_phrase}"
        f"{remark_phrase}{power_phrase}{proto_phrase}",
        f"create a {distro_phrase} VM named {name} with {vcpu} vCPU {mem}G ram "
        f"{disk}G disk{remark_phrase}{power_phrase}{proto_phrase}",
        f"开机器：{name}, {distro_phrase}, {vcpu}c{mem}g{disk}g"
        f"{remark_phrase}{power_phrase}{proto_phrase}",
    ]
    user = random.choice(templates)
    return _one_call_sample(user, "create_vm", args)


# ---- multi-call samples (一条消息里多个 create_vm) -------------------------
def sample_multi_create_vm() -> dict:
    n = random.randint(2, 3)
    same_spec = random.random() < 0.5  # 一半的样本是同规格批量, 另一半各异
    common_os, common_distro = _rand_os()
    common_vcpu, common_mem, common_disk = _rand_vcpu(), _rand_mem(), _rand_disk()

    name_roots = random.sample(NAME_ROOTS, n)
    names = [f"{r}-{random.randint(1, 9)}" for r in name_roots]
    vms = []
    for nm in names:
        if same_spec:
            vms.append({
                "vm_name": nm, "vcpu": common_vcpu, "memory_gb": common_mem,
                "disk_gb": common_disk, "os_type": common_os,
            })
        else:
            os_type, _ = _rand_os()
            vms.append({
                "vm_name": nm, "vcpu": _rand_vcpu(), "memory_gb": _rand_mem(),
                "disk_gb": _rand_disk(), "os_type": os_type,
            })

    # 拼用户语言
    if same_spec:
        templates = [
            f"开 {n} 台 {common_vcpu}c{common_mem}g {common_distro} 虚机，"
            f"分别叫 {'、'.join(names)}，磁盘都 {common_disk}G",
            f"批量建 {n} 台 {common_distro}，{common_vcpu} 核 {common_mem}G "
            f"内存 {common_disk}G 盘，名字是 {'、'.join(names)}",
            f"create {n} {common_distro} VMs: " + ", ".join(names) +
            f", all {common_vcpu}c {common_mem}G {common_disk}G",
        ]
    else:
        parts = []
        for v in vms:
            distro_label = "windows" if v["os_type"] == OS_TYPE_WINDOWS else "linux"
            parts.append(
                f"{v['vm_name']}({v['vcpu']}c{v['memory_gb']}g"
                f"{v['disk_gb']}g {distro_label})"
            )
        templates = [
            "帮我开这几台机器：" + "、".join(parts),
            "建以下虚机：" + "; ".join(parts),
            "create these VMs: " + ", ".join(parts),
        ]

    tool_calls = [
        {"type": "function", "function": {"name": "create_vm", "arguments": args}}
        for args in vms
    ]
    return {
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": random.choice(templates)},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ],
    }


# ---- negative samples (don't call any tool) -------------------------------
NEGATIVE_PAIRS = [
    ("你是谁？", "我是 HCI 平台的运维助手，可以帮你创建虚拟机。"),
    ("hi", "你好，需要我帮你建机器吗？"),
    ("Kubernetes 和 Docker 啥区别？",
     "Docker 是容器运行时，Kubernetes 是容器编排平台；这跟我能做的虚拟机创建是两套不同的东西。"),
    ("帮我写个冒泡排序",
     "这个问题与云平台无关，我只能帮你在 HCI 上创建虚拟机；如果你想在某台虚机里跑代码，可以先让我建一台。"),
    ("今天天气怎么样？", "我不能查天气，只能帮你建虚机。"),
    ("KVM 和 VMware 哪个好？",
     "都是主流虚拟化方案。我们这套用的是 KVM；如果你要在这个平台上建机器我可以直接帮你做。"),
    ("spice 是啥", "spice 是一种远程桌面协议，对延迟敏感的图形场景比 vnc 体验更好；HCI 这里默认就是 spice。"),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg_ratio", type=float, default=0.08)
    ap.add_argument("--multi_call_ratio", type=float, default=0.20,
                    help="比例: 一条消息内多次 tool_call")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_neg = int(args.n * args.neg_ratio)
    n_multi = int(args.n * args.multi_call_ratio)
    n_single = args.n - n_neg - n_multi

    samples: list[dict] = []
    for _ in range(n_single):
        samples.append(sample_create_vm())
    for _ in range(n_multi):
        samples.append(sample_multi_create_vm())
    for _ in range(n_neg):
        samples.append(sample_negative())

    random.shuffle(samples)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"wrote {len(samples)} samples → {out_path}")
    print(f"  single-call: {n_single}  multi-call: {n_multi}  negative: {n_neg}")


if __name__ == "__main__":
    main()
