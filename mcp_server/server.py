"""MCP server (stdio) exposing the same cloud tools.

跑起来后，可以在 Claude Desktop / Claude Code 的 MCP 配置里挂上它：

    {
      "mcpServers": {
        "cloud-poc": {
          "command": "python",
          "args": ["/abs/path/to/mcp_server/server.py"]
        }
      }
    }

或用 `mcp dev mcp_server/server.py` 启动调试 UI。

注意：MCP server 与微调后的 Qwen 是解耦的 — 任何兼容 MCP 的客户端都能调
同一套工具。这样可以方便地横向对比：
    - 微调 Qwen + 自家 agent_loop
    - 任意大模型 + MCP 客户端
对同一组工具的调用稳定性差异。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.cloud_tools import TOOLS, dispatch  # noqa: E402

mcp = FastMCP("cloud-poc")


def _register_tool(spec: dict) -> None:
    """Register one OpenAI-style tool spec onto FastMCP dynamically."""
    fn_spec = spec["function"]
    name = fn_spec["name"]
    description = fn_spec.get("description", "")
    schema = fn_spec.get("parameters", {"type": "object", "properties": {}})

    async def _handler(**kwargs):  # noqa: ANN001
        result = dispatch(name, kwargs)
        return json.dumps(result, ensure_ascii=False)

    # FastMCP 默认从 type hints 推 schema；我们绕过它，直接注入 JSON schema
    _handler.__name__ = name
    _handler.__doc__ = description
    mcp.add_tool(_handler, name=name, description=description, schema=schema)


for spec in TOOLS:
    _register_tool(spec)


if __name__ == "__main__":
    # stdio transport: 默认行为，适合 Claude Desktop / Claude Code 拉起
    mcp.run()
