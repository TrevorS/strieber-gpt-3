#!/usr/bin/env python3
# ABOUTME: CLI tool for testing MCP servers interactively.
# ABOUTME: Supports listing tools and calling them with JSON arguments.

import argparse
import asyncio
import json
import sys
from typing import Any

from common.mcp_client import MCPClient, MCPError, MCPTool


def format_tool(tool: MCPTool, verbose: bool = False) -> str:
    """Format a tool for display."""
    lines = [f"  {tool.name}"]
    if tool.description:
        # Truncate long descriptions
        desc = tool.description
        if len(desc) > 80 and not verbose:
            desc = desc[:77] + "..."
        lines.append(f"    {desc}")
    if verbose and tool.inputSchema:
        props = tool.inputSchema.get("properties", {})
        required = set(tool.inputSchema.get("required", []))
        if props:
            lines.append("    Parameters:")
            for name, schema in props.items():
                req = "*" if name in required else ""
                ptype = schema.get("type", "any")
                pdesc = schema.get("description", "")
                if pdesc:
                    lines.append(f"      {name}{req} ({ptype}): {pdesc[:60]}")
                else:
                    lines.append(f"      {name}{req} ({ptype})")
    return "\n".join(lines)


def format_result(result: Any) -> str:
    """Format a tool result for display."""
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)


async def cmd_list(args: argparse.Namespace) -> int:
    """List tools on a server."""
    try:
        client = MCPClient(args.server, timeout=args.timeout)
        tools = await client.list_tools()

        if not tools:
            print(f"No tools found on {client.url}")
            return 0

        print(f"Tools on {client.url}:\n")
        for tool in tools:
            print(format_tool(tool, verbose=args.verbose))
            print()

        return 0

    except MCPError as e:
        print(f"MCP Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def cmd_call(args: argparse.Namespace) -> int:
    """Call a tool on a server."""
    try:
        client = MCPClient(args.server, timeout=args.timeout)

        # Parse arguments
        if args.args:
            try:
                arguments = json.loads(args.args)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON arguments: {e}", file=sys.stderr)
                return 1
        else:
            arguments = {}

        # Add any --key=value arguments
        for kv in args.kwargs or []:
            if "=" in kv:
                key, value = kv.split("=", 1)
                # Try to parse as JSON, otherwise use as string
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value

        print(f"Calling {args.tool} on {client.url}")
        if arguments:
            print(f"Arguments: {json.dumps(arguments, indent=2)}")
        print()

        result = await client.call_tool(args.tool, arguments)

        if result.is_error:
            print("ERROR:", file=sys.stderr)
            print(result.text(), file=sys.stderr)
            return 1

        # Display content
        for item in result.content:
            item_type = item.get("type", "unknown")
            if item_type == "text":
                print(item.get("text", ""))
            elif item_type == "image":
                # Show image metadata, not full base64
                data = item.get("data", "")
                mime = item.get("mimeType", "image/png")
                print(f"[Image: {mime}, {len(data)} bytes base64]")
            else:
                print(f"[{item_type}]: {json.dumps(item, indent=2)}")

        return 0

    except MCPError as e:
        print(f"MCP Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def cmd_ping(args: argparse.Namespace) -> int:
    """Check if a server is reachable."""
    try:
        client = MCPClient(args.server, timeout=5.0)
        if await client.ping():
            print(f"✓ {client.url} is reachable")
            return 0
        else:
            print(f"✗ {client.url} is not reachable")
            return 1
    except Exception as e:
        print(f"✗ {args.server}: {e}")
        return 1


async def cmd_servers(args: argparse.Namespace) -> int:
    """List known server aliases."""
    print("Known MCP servers:\n")
    for alias, url in MCPClient.SERVERS.items():
        print(f"  {alias:20} {url}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Test MCP servers interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s servers                          # List known server aliases
  %(prog)s list weather                     # List tools on weather server
  %(prog)s list http://localhost:9100/mcp   # List tools by URL
  %(prog)s call weather get_weather '{"location": "NYC"}'
  %(prog)s call weather get_weather location=NYC
  %(prog)s ping weather                     # Check if server is up
        """,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # servers command
    servers_parser = subparsers.add_parser("servers", help="List known server aliases")
    servers_parser.set_defaults(func=cmd_servers)

    # list command
    list_parser = subparsers.add_parser("list", help="List tools on a server")
    list_parser.add_argument("server", help="Server URL or alias (e.g., weather)")
    list_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed parameter info"
    )
    list_parser.set_defaults(func=cmd_list)

    # call command
    call_parser = subparsers.add_parser("call", help="Call a tool")
    call_parser.add_argument("server", help="Server URL or alias")
    call_parser.add_argument("tool", help="Tool name to call")
    call_parser.add_argument(
        "args", nargs="?", default=None, help="JSON arguments (optional)"
    )
    call_parser.add_argument(
        "kwargs",
        nargs="*",
        metavar="key=value",
        help="Additional arguments as key=value pairs",
    )
    call_parser.set_defaults(func=cmd_call)

    # ping command
    ping_parser = subparsers.add_parser("ping", help="Check if server is reachable")
    ping_parser.add_argument("server", help="Server URL or alias")
    ping_parser.set_defaults(func=cmd_ping)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
