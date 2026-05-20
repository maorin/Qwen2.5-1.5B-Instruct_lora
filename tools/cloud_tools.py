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

import os
import uuid
from typing import Any, Callable

import requests


# ---- env-specific defaults (来自用户提供的 192.168.7.91 集群样例) ----------
#
# 注意区分 *两个* IP（容易混淆）：
#   api_base:  HTTP 请求实际发到的入口 —— 一般是 cluster VIP (traefik 80 端口)
#   host_ip:   body 里 target host —— cluster 内具体目标节点 (VM 落在哪)
DEFAULT_ENV: dict[str, Any] = {
    # api_base：hci_copilot 调 HCI API 走的入口
    # 2026-05-19：用户开放 traefik :8006 (theapi entrypoint) 直连 hci_api 微服务，
    # 此端口 path 没有 /theapi/ 前缀（traefik path-strip 已在 router 层做）。
    # 接口路径全部以 /v5/ 开头：
    #   - POST /v5/vm/create
    #   - GET  /v5/storage/pool/volume/list
    # env HCI_API_BASE_URL 可覆盖。
    "api_base": "http://192.168.7.90:8006",
    "host_ip": "192.168.7.91",            # body.host_ip：VM 创建目标节点
    "host_pid": "b7e08766-f1f9-40f3-9887-a96d3ae3b1a8",
    "pool": "default_pool",
    "pool_id": "e31c06a3-0132-4145-a976-0402f99f7e07",
    "path_prefix": "/hcidata/default_pool/volumes",
    "network": "port1",
    "net_id": "f96dff9f-20b1-4907-b9eb-6c828649f769",
    "switch_id": "56a3b123-9b1e-4701-9143-71f050ac8c13",
    # 2026-05-19：手工映射表，作为 list 接口拿不到 (cookie/csrf 未通) 时的 fallback。
    # 数据来源：实测 GET /theapi/v5/storage/pool/volume/list 的空闲 (domain_count=0) 卷。
    # 后续 list 走通后可弃用本表。
    "volume_id_by_name": {
        "beijing.qcow2":  "904d560a-6c7d-4b68-ac25-8f0b2a3dbe66",
        "beijing1.qcow2": "e3bbdfc4-ae62-4389-bf22-4351483a831d",
        "beijing2.qcow2": "22665d47-5a87-40b8-b5a3-4096f184d6b6",
        "beijing3.qcow2": "88516ab1-a5ae-48d5-b44d-900123edf619",
        "beijing4.qcow2": "04e9fec5-ebf0-4d4b-92bb-02cd9edc5805",
        "beijing5.qcow2": "388cc116-d2d9-484d-bf6d-3c96d1edf568",
    },
    # 2026-05-20：/vm/create/use/template 端点用的 host_id 与 /vm/create 的
    # host_pid 是 *两个不同的 UUID*（来自用户提供的真实 curl 示例）。
    # 必须分开维护，不能复用 host_pid。
    "template_host_id": "06c39f1a-b4d4-45be-a8a6-cb1ea168fd91",
    # 2026-05-20：模板 UUID 映射，作为 /v5/vm/template/list 拿不到时的 fallback。
    # 当前仅有用户 curl 示例给的一个 UUID，归到 "default"；
    # 后续从 list 接口拿到真实模板名再填进来。
    "template_id_by_name": {
        "default": "d48ef798-710a-442d-9e71-8041fc208a0e",
    },
}


# ---- HTTP 调用配置 (2026-05-19) -------------------------------------------
# 通过环境变量覆盖避免改代码：
#   HCI_API_BASE_URL      API 基地址 (默认走 DEFAULT_ENV["api_base"] = traefik VIP)
#   HCI_API_TOKEN         Bearer token (若 hci_api 后端开了 Bearer 认证)
#   HCI_DRY_RUN=1         跳过真发请求，仅返回拼好的 body (训练数据生成 / demo)
#   HCI_API_VERIFY_SSL    默认 0 (hci 私网集群一般 traefik 自签名 cert)。
#                         生产用 CA-signed cert 时 set 1 强制校验。
#
# hci_auth 走 Cookie session (username + sessionid)，两种注入方式任选：
#   HCI_API_COOKIES       完整 Cookie header 字符串 'username=admin; sessionid=xx'
#   HCI_SESSION_USERNAME  + HCI_SESSION_ID 命名独立 env (更可读)
def _api_base_url() -> str:
    if url := os.environ.get("HCI_API_BASE_URL"):
        return url.rstrip("/")
    return DEFAULT_ENV["api_base"].rstrip("/")


def _is_dry_run() -> bool:
    return os.environ.get("HCI_DRY_RUN", "").lower() in ("1", "true", "yes", "on")


def _verify_ssl() -> bool:
    return os.environ.get("HCI_API_VERIFY_SSL", "0").lower() in ("1", "true", "yes", "on")


def _api_cookies() -> dict[str, str]:
    """从 env 构造 cookies dict (hci_auth 用 username + sessionid)。"""
    cookies: dict[str, str] = {}
    if raw := os.environ.get("HCI_API_COOKIES"):
        for part in raw.split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                cookies[k.strip()] = v.strip()
    if u := os.environ.get("HCI_SESSION_USERNAME"):
        cookies["username"] = u
    if s := os.environ.get("HCI_SESSION_ID"):
        cookies["sessionid"] = s
    return cookies


def _csrf_headers(cookies: dict[str, str] | None) -> dict[str, str]:
    """从 cookies 提 _xsrf 自动注入 CSRF header（Tornado 风格）。

    后端规则：cookie._xsrf 必须与 header.X-Xsrftoken 匹配才放行。
    同时也带 X-CSRFToken 兼容 Django / 其他框架的命名。
    """
    if not cookies:
        return {}
    xsrf = cookies.get("_xsrf") or cookies.get("csrftoken") or cookies.get("XSRF-TOKEN")
    if not xsrf:
        return {}
    return {"X-Xsrftoken": xsrf, "X-CSRFToken": xsrf}


def _default_browser_headers() -> dict[str, str]:
    """浏览器跑 vue3 dev 时一般带的头，list 接口可能查 Origin/Referer 做 CSRF。"""
    return {
        "Origin": os.environ.get("HCI_API_ORIGIN", "http://localhost:3000"),
        "Referer": os.environ.get("HCI_API_REFERER", "http://localhost:3000/"),
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json",
    }


def _get_from_hci(
    path: str,
    timeout: float = 10.0,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """通用 GET 辅助，跟 _post_to_hci 共用 base_url / cookies / ssl 配置。"""
    if _is_dry_run():
        return {"dry_run": True, "success": True, "response": []}
    url = f"{_api_base_url()}{path}"
    verify = _verify_ssl()
    final_cookies = cookies if cookies else _api_cookies()

    headers = {**_default_browser_headers(), **_csrf_headers(final_cookies)}

    print(f"[_get_from_hci] GET {url}", flush=True)
    print(f"[_get_from_hci]   cookies keys: {list((final_cookies or {}).keys())}", flush=True)
    print(f"[_get_from_hci]   headers: {list(headers.keys())}", flush=True)

    try:
        resp = requests.get(
            url, timeout=timeout, verify=verify,
            cookies=final_cookies or None, headers=headers,
        )
    except requests.RequestException as e:
        return {"success": False, "status_code": None, "error": f"{type(e).__name__}: {e}"}

    print(f"[_get_from_hci]   ← status {resp.status_code} body[:200]={resp.text[:200]!r}", flush=True)

    out: dict[str, Any] = {"status_code": resp.status_code, "success": 200 <= resp.status_code < 300}
    try:
        out["response"] = resp.json()
    except ValueError:
        out["response"] = resp.text[:500]
    if not out["success"]:
        out["error"] = f"HTTP {resp.status_code}"
    return out


def _resolve_disk_id(
    vm_name: str,
    cookies: dict[str, str] | None,
    pool_id: str | None = None,
) -> tuple[str, str]:
    """从 /storage/pool/volume/list 找一个**可用的**空闲 volume，用作 disk_id。

    list 返回 schema (实测 2026-05-19)：
        {"code":"200","msg":"ok","data":[
            {"id":"…","name":"beijing5.qcow2","storage_pool_id":"…",
             "domain_count":0,        # 0 = 空闲，≥1 = 已挂载到 VM
             "domain":null,           # null 同上
             "volume_type":"qcow2","use_type":"data","capacity":..., ...},
            ...
        ]}

    选择优先级（保守：宁报错也不用错卷）：
      1. name == '{vm_name}.qcow2' 且 domain_count==0 且 (pool_id 匹配或未传)
         → matched_idle_name
      2. 任意空闲 volume（domain_count==0）且 name 跟 vm_name 前缀同 (vm_name in name)
         → matched_idle_partial
      3. 全部空闲 volume 的第一个
         → first_idle (no name match)
      4. 都失败 → 随机 UUID，让 backend 报清晰错（不会误用已挂载卷）
         → fallback_uuid_no_idle

    返回 (disk_id, source) — source 记录决策路径便于 LLM/用户调试。
    """
    target = f"{vm_name}.qcow2"

    # 0. 先看 hardcoded 映射表是否有精确匹配（list 接口暂时拿不到时的 fast-path）
    hardcoded: dict[str, str] = DEFAULT_ENV.get("volume_id_by_name", {})
    if target in hardcoded:
        return hardcoded[target], f"hardcoded_map={target}"

    res = _get_from_hci("/v5/storage/pool/volume/list", cookies=cookies)
    if not res.get("success"):
        # list 拿不到 → 用 hardcoded 表第一项作 fallback（远好于随机 UUID）
        if hardcoded:
            name, vid = next(iter(hardcoded.items()))
            return vid, f"hardcoded_fallback={name} (list failed: {res.get('error')})"
        return str(uuid.uuid4()), f"fallback_uuid (list failed: {res.get('error')}, no hardcoded map)"

    data = res.get("response")
    # 兼容多种包裹：[...] / {data:[...]} / {volumes:[...]} / {list:[...]} / {results:[...]}
    volumes = data if isinstance(data, list) else (
        (data or {}).get("data") or (data or {}).get("volumes")
        or (data or {}).get("list") or (data or {}).get("results") or []
    )
    if not isinstance(volumes, list):
        return str(uuid.uuid4()), f"fallback_uuid (unexpected schema: {type(data).__name__})"

    def is_idle(v: dict) -> bool:
        # domain_count==0 且 domain 为 null/空 → 未挂载到任何 VM
        if v.get("domain_count", 0) > 0:
            return False
        d = v.get("domain")
        return not d  # null / [] / {} 都算空闲

    def pool_ok(v: dict) -> bool:
        if not pool_id:
            return True
        spi = v.get("storage_pool_id") or v.get("pool_id")
        return not spi or spi == pool_id

    def vol_id(v: dict) -> str | None:
        return v.get("id") or v.get("disk_id") or v.get("volume_id") or v.get("vol_id")

    idle = [v for v in volumes if isinstance(v, dict) and is_idle(v) and pool_ok(v)]

    # 1. 精确 name 匹配 + 空闲
    for v in idle:
        name = v.get("name") or v.get("volume_name") or v.get("disk_name")
        if name == target:
            if vid := vol_id(v):
                return vid, f"matched_idle_name={target}"

    # 2. 包含 vm_name 前缀 + 空闲（容忍 list 里把 vm_name 拼成别的后缀）
    for v in idle:
        name = v.get("name") or v.get("volume_name") or v.get("disk_name") or ""
        if vm_name and vm_name in name:
            if vid := vol_id(v):
                return vid, f"matched_idle_partial={name}"

    # 3. 任意空闲 volume
    if idle:
        v = idle[0]
        if vid := vol_id(v):
            n = v.get("name") or v.get("volume_name") or "?"
            return vid, f"first_idle={n} (total_idle={len(idle)})"

    # 4. list OK 但没空闲 volume → fallback 到 hardcoded（避免随机 UUID）
    used_count = sum(1 for v in volumes if isinstance(v, dict) and not is_idle(v))
    if hardcoded:
        name, vid = next(iter(hardcoded.items()))
        return vid, f"hardcoded_fallback={name} (list ok but no idle; total={len(volumes)} used={used_count})"
    return (
        str(uuid.uuid4()),
        f"fallback_uuid_no_idle (total={len(volumes)} used={used_count} target_name={target})",
    )


def _resolve_template_id(
    template: str | None,
    cookies: dict[str, str] | None,
) -> tuple[str, str]:
    """Resolve human-friendly template name → template_id UUID.

    Strategy 同 _resolve_disk_id：
      1. 精确匹配 hardcoded `template_id_by_name`
      2. 调 list 接口（猜测路径 /v5/vm/template/list）找 name 匹配
      3. list 拿到模板但没匹配 name → 返回第一个模板
      4. list 失败 → hardcoded 第一个作为兜底
      5. 全失败 → 随机 UUID（backend 会报清晰错，不会误用错模板）

    返回 (template_id, source) — source 记录决策路径便于调试。
    """
    hardcoded: dict[str, str] = DEFAULT_ENV.get("template_id_by_name", {})
    key = (template or "default").strip()

    # 1. hardcoded 精确匹配
    if key in hardcoded:
        return hardcoded[key], f"hardcoded_map={key}"

    # 2. 尝试 list 接口
    res = _get_from_hci("/v5/vm/template/list", cookies=cookies)
    if not res.get("success"):
        if hardcoded:
            name, tid = next(iter(hardcoded.items()))
            return tid, f"hardcoded_fallback={name} (list failed: {res.get('error')})"
        return str(uuid.uuid4()), f"fallback_uuid (list failed: {res.get('error')})"

    data = res.get("response")
    templates = data if isinstance(data, list) else (
        (data or {}).get("data") or (data or {}).get("templates")
        or (data or {}).get("list") or (data or {}).get("results") or []
    )
    if not isinstance(templates, list):
        return str(uuid.uuid4()), f"fallback_uuid (unexpected schema: {type(data).__name__})"

    def tid_of(t: dict) -> str | None:
        return t.get("id") or t.get("template_id") or t.get("uuid")

    # 2a. name 精确匹配
    for t in templates:
        if isinstance(t, dict):
            name = t.get("name") or t.get("template_name") or ""
            if name == key:
                if tid := tid_of(t):
                    return tid, f"matched_name={name}"

    # 2b. 任一模板包含 key 子串
    for t in templates:
        if isinstance(t, dict):
            name = t.get("name") or t.get("template_name") or ""
            if key and key.lower() in name.lower():
                if tid := tid_of(t):
                    return tid, f"matched_partial={name}"

    # 2c. 返回第一个可用模板
    for t in templates:
        if isinstance(t, dict):
            if tid := tid_of(t):
                n = t.get("name") or "?"
                return tid, f"first_template={n} (total={len(templates)}, no name match for '{key}')"

    # 3. list 成功但没拿到任何模板 → hardcoded fallback
    if hardcoded:
        name, tid = next(iter(hardcoded.items()))
        return tid, f"hardcoded_fallback={name} (list ok but empty)"
    return str(uuid.uuid4()), "fallback_uuid (list ok but empty, no hardcoded)"


def _post_to_hci(
    path: str,
    body: dict[str, Any],
    timeout: float = 30.0,
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """真实 POST 到 HCI /theapi/* 接口。

    返回字段（merge 进 _create_vm 返回值，LLM 会看到并总结给用户）：
        success:     bool                       — 总开关
        status_code: int | None                 — HTTP 状态码，超时/连接失败为 None
        response:    dict | str                 — 解析后的 JSON 或前 500 字符文本
        error:       str (仅失败时)             — 异常类型 + 消息

    HCI_DRY_RUN=1 时跳过网络请求，返回 {"dry_run": True, "success": True}。
    """
    if _is_dry_run():
        return {"dry_run": True, "success": True}

    url = f"{_api_base_url()}{path}"
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("HCI_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    verify = _verify_ssl()
    if not verify:
        # 抑制 traefik 自签名 cluster 的 InsecureRequestWarning 刷屏
        try:
            from urllib3.exceptions import InsecureRequestWarning  # noqa: WPS433
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Cookies 优先级：传入参数 > env (HCI_API_COOKIES / HCI_SESSION_*)。前者用于
    # 转发 hci_vue3 浏览器 cookie；后者用于 server-side 配置（dev / 批处理）。
    final_cookies = cookies if cookies else _api_cookies()

    try:
        resp = requests.post(
            url, json=body, headers=headers,
            timeout=timeout, verify=verify, cookies=final_cookies or None,
        )
    except requests.RequestException as e:
        return {
            "success": False,
            "status_code": None,
            "error": f"{type(e).__name__}: {e}",
        }

    out: dict[str, Any] = {
        "status_code": resp.status_code,
        "success": 200 <= resp.status_code < 300,
    }
    try:
        out["response"] = resp.json()
    except ValueError:
        out["response"] = resp.text[:500]
    if not out["success"]:
        out["error"] = f"HTTP {resp.status_code}"
    return out

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
                "POST /v5/vm/create (经 traefik theapi entrypoint :8006 直连)。"
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
    {
        "type": "function",
        "function": {
            "name": "create_vm_from_template",
            "description": (
                "从已有模板克隆一台 KVM 虚拟机。对应真实接口 "
                "POST /v5/vm/create/use/template。"
                "用户明确提到 \"模板\" / \"template\" 时优先用这个 (规格继承自模板, "
                "无需 vcpu/memory/disk); 如果用户报了 CPU/内存/磁盘等具体规格, "
                "改用 create_vm。"
                "环境字段 (template_id / host_id) 由平台 agent 解析, 你只填模板名。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vm_name": {
                        "type": "string",
                        "description": "新虚拟机名称, 集群内唯一",
                    },
                    "template": {
                        "type": "string",
                        "description": (
                            "模板名 (如 'ubuntu-22.04', 'kylin-desktop', 'windows-server-2022', "
                            "'default')。平台 agent 会查模板列表把名字解析成 template_id UUID。"
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "同一模板批量创建几台 (端点原生支持); 默认 1",
                        "minimum": 1, "maximum": 50, "default": 1,
                    },
                    "remark": {
                        "type": "string",
                        "description": "备注信息, 没有可不填",
                    },
                },
                "required": ["vm_name", "template"],
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
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Expand LLM-friendly args into the real HCI /vm/create body."""
    disk_name = f"{vm_name}.qcow2"
    # 2026-05-19：先查 volume list 决定 disk_id。后端按 disk_id 查 volume，
    # 旧版用随机 UUID 导致 'NoneType' has no attribute 'volume_type' 500 错。
    disk_id, disk_id_source = _resolve_disk_id(
        vm_name, cookies, pool_id=DEFAULT_ENV["pool_id"],
    )
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
            "disk_id": disk_id,
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
    result: dict[str, Any] = {
        "endpoint": "POST /v5/vm/create",
        "host": DEFAULT_ENV["host_ip"],
        "body": body,
        "disk_id_source": disk_id_source,  # 调试：disk_id 是匹配 / 列表首项 / fallback
    }
    # 2026-05-19：真发 HTTP POST，把响应/错误合并回 result 给 LLM 总结
    result.update(_post_to_hci("/v5/vm/create", body, cookies=cookies))
    return result


def _create_vm_from_template(
    vm_name: str,
    template: str,
    count: int = 1,
    remark: str = "",
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Expand LLM-friendly args into the real /v5/vm/create/use/template body.

    与 _create_vm 的关键差别 (易踩坑, 已加注释提醒):
      - 用 host_id (template_host_id), 不是 host_pid
      - 没有 disk/interface/cdrom 嵌套数组, body 扁平 5 字段
      - 模板已经决定了 CPU/内存/磁盘/网络, 这里不暴露这些字段
    """
    template_id, template_id_source = _resolve_template_id(template, cookies)
    body = {
        "template_id": template_id,
        "host_id": DEFAULT_ENV["template_host_id"],
        "vm_name": vm_name,
        "count": count,
        "remark": remark,
    }
    result: dict[str, Any] = {
        "endpoint": "POST /v5/vm/create/use/template",
        "host": DEFAULT_ENV["host_ip"],
        "body": body,
        "template_id_source": template_id_source,
    }
    result.update(_post_to_hci("/v5/vm/create/use/template", body, cookies=cookies))
    return result


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "create_vm": _create_vm,
    "create_vm_from_template": _create_vm_from_template,
}


def dispatch(
    name: str,
    arguments: dict[str, Any],
    cookies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a tool by name. Returns a JSON-serializable dict.

    与 OpenAI / Anthropic function-calling 一致, 剔掉 None 值再分发,
    宽容地处理小模型偶尔的 schema bleeding。

    cookies (2026-05-19 加)：会话 cookie，hci_copilot server 把浏览器请求带的
    cookie 透传过来给本函数；本函数 inspect handler 签名，若 handler 接 cookies
    参数则注入。LLM 不感知，cookies 不会进 LLM context。
    """
    if name not in _HANDLERS:
        return {"error": f"unknown tool: {name}"}
    cleaned = {k: v for k, v in (arguments or {}).items() if v is not None}
    handler = _HANDLERS[name]
    # 透传 cookies 给愿意接收的 handler（_create_vm 等）
    import inspect
    if cookies is not None and "cookies" in inspect.signature(handler).parameters:
        cleaned["cookies"] = cookies
    try:
        return handler(**cleaned)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# legacy hooks (data/gen_dataset.py 老接口需要), 在 create-only 模式下退化为 no-op
def reset_state() -> None:
    pass


def snapshot_state() -> dict[str, Any]:
    return {}
