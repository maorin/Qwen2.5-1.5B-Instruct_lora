"""Single source of truth for the HCI create_vm tool.

设计原则 (production-style facade):
- 给 LLM 暴露的是 **简化 schema** (8 个核心字段, 都是 LLM 能从用户原话推出来的);
- dispatch 在 Python 侧用 DEFAULT_ENV 把 host_pid / pool_id / net_id / switch_id /
  path 等 env-specific 字段补齐, 拼出真实 HCI 接口 POST /theapi/v5/vm/create
  能接受的 30+ 字段 payload;
- 不让 1.5B 模型直接生成 UUID 或重复 boilerplate。

如果以后接多租户 / 多集群, 把 DEFAULT_ENV 改成从 config 读即可, 上层 schema
和训练数据都不用动。
"""
from __future__ import annotations

import uuid
from typing import Any, Callable


# ---- env-specific defaults (来自用户提供的 192.168.7.91 集群样例) ----------
DEFAULT_ENV: dict[str, Any] = {
    "host_ip": "192.168.7.91",
    "host_pid": "b7e08766-f1f9-40f3-9887-a96d3ae3b1a8",
    "pool": "default_pool",
    "pool_id": "e31c06a3-0132-4145-a976-0402f99f7e07",
    "path_prefix": "/hcidata/default_pool/volumes",
    "network": "port1",
    "net_id": "f96dff9f-20b1-4907-b9eb-6c828649f769",
    "switch_id": "56a3b123-9b1e-4701-9143-71f050ac8c13",
}

# os_type 整数编码 (HCI 常见约定; 用户未明确, 如不符可改)
OS_TYPE_LINUX = 1
OS_TYPE_WINDOWS = 2


# ---- LLM 看的简化 schema ---------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_vm",
            "description": (
                "在 HCI 平台上创建一台 KVM 虚拟机。对应真实接口 "
                "POST /theapi/v5/vm/create。"
                "环境相关字段 (host_pid / pool_id / net_id / switch_id / 路径等) "
                "由平台 agent 自动补齐, 你不需要填。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vm_name": {
                        "type": "string",
                        "description": "虚拟机名称, 须在集群内唯一",
                    },
                    "vcpu": {
                        "type": "integer",
                        "description": "vCPU 核数",
                        "minimum": 1, "maximum": 128,
                    },
                    "memory_gb": {
                        "type": "integer",
                        "description": "内存大小, 单位 GB",
                        "minimum": 1, "maximum": 1024,
                    },
                    "disk_gb": {
                        "type": "integer",
                        "description": "系统盘大小, 单位 GB",
                        "minimum": 10, "maximum": 8192,
                    },
                    "os_type": {
                        "type": "integer",
                        "enum": [OS_TYPE_LINUX, OS_TYPE_WINDOWS],
                        "description": "操作系统类型: 1=Linux 家族 (含 ubuntu/centos/debian/rocky/kylin/uos), 2=Windows",
                    },
                    "remark": {
                        "type": "string",
                        "description": "备注信息, 没有可不填",
                    },
                    "display_protocol": {
                        "type": "string",
                        "enum": ["spice", "vnc"],
                        "description": "图形协议, 默认 spice",
                    },
                    "power_on": {
                        "type": "boolean",
                        "description": "创建后是否立即开机, 默认 false",
                    },
                },
                "required": ["vm_name", "vcpu", "memory_gb", "disk_gb", "os_type"],
            },
        },
    },
]


# ---- dispatch: 简化字段 → 真实 HCI payload ---------------------------------
def _create_vm(
    vm_name: str,
    vcpu: int,
    memory_gb: int,
    disk_gb: int,
    os_type: int,
    remark: str = "",
    display_protocol: str = "spice",
    power_on: bool = False,
) -> dict[str, Any]:
    """Expand LLM-friendly args into the real HCI /vm/create body."""
    disk_name = f"{vm_name}.qcow2"
    body = {
        "host_ip": DEFAULT_ENV["host_ip"],
        "host_pid": DEFAULT_ENV["host_pid"],
        "vm_name": vm_name,
        "os_type": os_type,
        "virtualization": "kvm",
        "remark": remark,
        "display_protocol": display_protocol,
        "allowmc": "0",
        "safe_status": 0,
        "open": "1" if power_on else "0",
        "vcpu_unit": str(vcpu),
        "numa_cpu": [],
        "memory_unit": str(memory_gb),
        "memory_unit_type": "GB",
        "mem_strategy": "",
        "numa_mem": "",
        "input_devices": {"mouse": "usb", "keyboard": "usb"},
        "disk": [{
            "pool": DEFAULT_ENV["pool"],
            "disk_type": "file",
            "size": str(disk_gb),
            "disk_unit_type": "GB",
            "path": f"{DEFAULT_ENV['path_prefix']}/{disk_name}",
            "disk_name": disk_name,
            "pool_id": DEFAULT_ENV["pool_id"],
            "disk_id": str(uuid.uuid4()),
        }],
        "interface": [{
            "interface_type": "network",
            "mac": "",
            "ip": "",
            "network": DEFAULT_ENV["network"],
            "net_id": DEFAULT_ENV["net_id"],
            "model": "virtio",
            "switch_id": DEFAULT_ENV["switch_id"],
        }],
        "cdrom": [{"disk_id": "", "pool_id": "", "storage_type_code": "", "path": ""}],
    }
    return {
        "endpoint": "POST /theapi/v5/vm/create",
        "host": DEFAULT_ENV["host_ip"],
        "body": body,
    }


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "create_vm": _create_vm,
}


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name. Returns a JSON-serializable dict.

    与 OpenAI / Anthropic function-calling 一致, 剔掉 None 值再分发,
    宽容地处理小模型偶尔的 schema bleeding。
    """
    if name not in _HANDLERS:
        return {"error": f"unknown tool: {name}"}
    cleaned = {k: v for k, v in (arguments or {}).items() if v is not None}
    try:
        return _HANDLERS[name](**cleaned)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# legacy hooks (data/gen_dataset.py 老接口需要), 在 create-only 模式下退化为 no-op
def reset_state() -> None:
    pass


def snapshot_state() -> dict[str, Any]:
    return {}
