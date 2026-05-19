"""MCP server (stdio) exposing the same cloud tools.

跑起来后，可以在 Claude Desktop / Claude Code 的 MCP 配置里挂上它：

    {
      "mcpServers": {
        "cloud-poc": {
          "command": "/opt/miniconda3/envs/qwen-cloud-poc/bin/python",
          "args": ["/abs/path/to/mcp_server/server.py"]
        }
      }
    }

注意：MCP server 与微调后的 Qwen 是解耦的 — 任何兼容 MCP 的客户端都能调
同一套工具。这样可以方便地横向对比：
    - 微调 Qwen + 自家 agent_loop
    - 任意大模型 + MCP 客户端
对同一组工具的调用稳定性差异。

使用 lowlevel Server (而不是 FastMCP)，因为 FastMCP.add_tool 是从函数的
type hints 反推 schema，无法直接喂 JSON schema；我们已经在
tools/cloud_tools.py 里维护了完整的 JSON schema，所以走 lowlevel 路径把
schema 原样透出最干净。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.cloud_tools import TOOLS, dispatch  # noqa: E402

server: Server = Server("cloud-poc")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=spec["function"]["name"],
            description=spec["function"].get("description", ""),
            inputSchema=spec["function"].get(
                "parameters", {"type": "object", "properties": {}}
            ),
        )
        for spec in TOOLS
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    result = dispatch(name, arguments or {})
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


async def probe() -> None:
    """In-process self-test: bypass MCP protocol, call handlers directly.
    只验证 dispatch 逻辑, 不验证 JSON-RPC 协议层。"""
    tools = await list_tools()
    print(f"=== {server.name}: {len(tools)} tools registered ===")
    for t in tools:
        required = t.inputSchema.get("required", [])
        props = list(t.inputSchema.get("properties", {}).keys())
        print(f"  • {t.name:14s} {t.description}")
        print(f"      params: {props}  required: {required}")

    print("\n=== sample call: create_vm ===")
    result = await call_tool(
        "create_vm",
        {"vm_name": "probe-vm", "vcpu": 4, "memory_gb": 8,
         "disk_gb": 50, "os_type": 1, "remark": "probe"},
    )
    print(result[0].text)


async def client_probe() -> None:
    """End-to-end probe: subprocess-spawn this same script as a stdio MCP
    server, then connect as a real MCP client and exercise the full
    JSON-RPC protocol (initialize → list_tools → call_tool)."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve())],  # 启动本脚本默认 (stdio server) 模式
    )

    print(f"=== spawning MCP server: {params.command} {params.args[0]}")
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print(f"=== handshake OK: server={init.serverInfo.name} "
                  f"v{init.serverInfo.version} protocol={init.protocolVersion}")

            tools = (await session.list_tools()).tools
            print(f"\n=== list_tools → {len(tools)} tools ===")
            for t in tools:
                print(f"  • {t.name:14s} required={t.inputSchema.get('required', [])}")

            print("\n=== call_tool create_vm via JSON-RPC ===")
            r = await session.call_tool("create_vm", {
                "vm_name": "client-probe", "vcpu": 4, "memory_gb": 8,
                "disk_gb": 50, "os_type": 1,
            })
            print(r.content[0].text)

            print("\n[ok] end-to-end MCP stdio protocol works.")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        anyio.run(probe)
    elif "--client-probe" in sys.argv:
        anyio.run(client_probe)
    else:
        anyio.run(main)
