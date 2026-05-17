"""Single source of truth for the mock cloud platform.

定义 OpenAI/Qwen 风格的 function-call JSON schema，并提供一个内存实现，
训练数据生成、推理 agent、MCP server 全部从此处派生 — 三处定义不会漂移。
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable

REGIONS = ["cn-east-1", "cn-east-2", "cn-north-1", "us-west-1", "ap-southeast-1"]
IMAGES = ["ubuntu-22.04", "ubuntu-20.04", "centos-7", "debian-12", "rocky-9"]
FLAVORS = ["1c2g", "2c4g", "4c8g", "8c16g", "16c32g"]
VM_STATUSES = ["pending", "running", "stopped", "error"]


# ---- schema (OpenAI function-calling style; Qwen tokenizer 直接吃) ----------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_vm",
            "description": "创建一台虚拟机。返回新建实例的 id 与初始状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "虚拟机名称，需在区域内唯一"},
                    "region": {"type": "string", "enum": REGIONS},
                    "image": {"type": "string", "enum": IMAGES},
                    "flavor": {"type": "string", "enum": FLAVORS, "description": "规格，如 4c8g 表示 4 核 8GB"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 20, "default": 1},
                },
                "required": ["name", "region", "image", "flavor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_vms",
            "description": "列出虚拟机，支持按 region/status 过滤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": REGIONS},
                    "status": {"type": "string", "enum": VM_STATUSES},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vm",
            "description": "查询单台虚拟机详细信息。",
            "parameters": {
                "type": "object",
                "properties": {"vm_id": {"type": "string"}},
                "required": ["vm_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_vm",
            "description": "停止指定虚拟机。",
            "parameters": {
                "type": "object",
                "properties": {"vm_id": {"type": "string"}},
                "required": ["vm_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_vm",
            "description": "启动已停止的虚拟机。",
            "parameters": {
                "type": "object",
                "properties": {"vm_id": {"type": "string"}},
                "required": ["vm_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_vm",
            "description": "删除虚拟机，操作不可恢复。",
            "parameters": {
                "type": "object",
                "properties": {"vm_id": {"type": "string"}},
                "required": ["vm_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": "查询虚拟机最近一段时间的 CPU/内存指标。",
            "parameters": {
                "type": "object",
                "properties": {
                    "vm_id": {"type": "string"},
                    "metric": {"type": "string", "enum": ["cpu", "memory", "disk_io"]},
                    "window_minutes": {"type": "integer", "minimum": 1, "maximum": 1440, "default": 15},
                },
                "required": ["vm_id", "metric"],
            },
        },
    },
]


# ---- in-memory backend ------------------------------------------------------
_VMS: dict[str, dict[str, Any]] = {}


def reset_state() -> None:
    _VMS.clear()


def snapshot_state() -> dict[str, Any]:
    return {"vms": list(_VMS.values())}


def _new_id() -> str:
    return "vm-" + uuid.uuid4().hex[:8]


def _create_vm(name: str, region: str, image: str, flavor: str, count: int = 1) -> dict[str, Any]:
    created = []
    for i in range(count):
        vid = _new_id()
        vm = {
            "vm_id": vid,
            "name": name if count == 1 else f"{name}-{i + 1}",
            "region": region,
            "image": image,
            "flavor": flavor,
            "status": "running",
            "created_at": int(time.time()),
        }
        _VMS[vid] = vm
        created.append(vm)
    return {"created": created}


def _list_vms(region: str | None = None, status: str | None = None) -> dict[str, Any]:
    out = list(_VMS.values())
    if region:
        out = [v for v in out if v["region"] == region]
    if status:
        out = [v for v in out if v["status"] == status]
    return {"vms": out, "count": len(out)}


def _get_vm(vm_id: str) -> dict[str, Any]:
    if vm_id not in _VMS:
        return {"error": f"vm {vm_id} not found"}
    return _VMS[vm_id]


def _set_status(vm_id: str, status: str) -> dict[str, Any]:
    if vm_id not in _VMS:
        return {"error": f"vm {vm_id} not found"}
    _VMS[vm_id]["status"] = status
    return {"vm_id": vm_id, "status": status}


def _delete_vm(vm_id: str) -> dict[str, Any]:
    if vm_id not in _VMS:
        return {"error": f"vm {vm_id} not found"}
    _VMS.pop(vm_id)
    return {"deleted": vm_id}


def _get_metrics(vm_id: str, metric: str, window_minutes: int = 15) -> dict[str, Any]:
    if vm_id not in _VMS:
        return {"error": f"vm {vm_id} not found"}
    import random
    random.seed(hash((vm_id, metric, window_minutes)) & 0xFFFFFFFF)
    points = [round(random.uniform(5, 95), 2) for _ in range(window_minutes)]
    return {"vm_id": vm_id, "metric": metric, "unit": "%", "points": points}


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "create_vm": _create_vm,
    "list_vms": _list_vms,
    "get_vm": _get_vm,
    "stop_vm": lambda vm_id: _set_status(vm_id, "stopped"),
    "start_vm": lambda vm_id: _set_status(vm_id, "running"),
    "delete_vm": _delete_vm,
    "get_metrics": _get_metrics,
}


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name. Returns a JSON-serializable dict.

    小模型偶尔会把别的工具的参数当作 null 一起填进来 (schema bleeding);
    与 OpenAI / Anthropic function-calling 行为一致, 这里先剔掉 None
    再分发, 让 dispatch 对噪声更宽容。
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
