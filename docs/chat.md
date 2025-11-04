Skip to content
Chat history


Invite team members
You said:
How do you add server side tools to open web ui backed by llama server
ChatGPT said:

Thought for 1m 14s
Short version:

Point Open WebUI at your llama.cpp server

Add your server-side tools (either built-in Python Tools or external OpenAPI tool servers)

Enable those tools for the model you’re chatting with and pick the function-calling mode.

Here’s the quick path:

1) Connect Open WebUI → llama.cpp (aka llama-server)
Start llama-server with your model, e.g.

./llama-server --model /path/model.gguf --port 10000
This exposes an OpenAI-compatible API at http://127.0.0.1:10000/v1. In Open WebUI: Admin Settings → Connections → OpenAI Connections → + Add, set the URL to your llama-server (use http://host.docker.internal:10000/v1 if WebUI runs in Docker).
docs.openwebui.com

2) Add “server-side tools” to Open WebUI
You’ve got two good options—use either or both:

A) Open WebUI Tools (Python, run on the WebUI server)

In WebUI, go to Features → Tools, then install from the community library or create your own.

To ship your own: make a single Python file with a Tools class and typed methods; WebUI turns that into callable tools. Example structure is shown in the docs (metadata docstring, Tools class, optional Valves for settings).
docs.openwebui.com
+1

B) External OpenAPI Tool Servers (FastAPI/any OpenAPI)

Start a tool server (e.g., the reference “time”, “filesystem”, etc.):

git clone https://github.com/open-webui/openapi-servers
cd openapi-servers/servers/time
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --reload
In WebUI Settings → Tools → + Tool Server, paste the tool server URL (if you’re proxying multiple with mcpo, use the exact subpath like /time or /memory).

Use User Tool Server for browser-local tools; use Global Tool Server in Admin Settings to make them available to everyone (requests originate from the WebUI backend).
docs.openwebui.com
+1

3) Enable tools for your llama.cpp-backed model
Go to Workspace → Models → (edit your model), check the tools you want available by default; or in chat click the ➕ next to the input to toggle tools for that conversation.
docs.openwebui.com

4) Choose tool-calling mode
In a chat: ⚙️ Chat Controls → Advanced Params → Function Calling.

Default: WebUI orchestrates tools via prompting (works with almost any local model).

Native: Passes OpenAI “tools” to the backend for the model to call functions natively (llama.cpp supports function calling; if it struggles to pick tools, switch back to Default).
docs.openwebui.com
+1

Notes & gotchas
Docker networking: If WebUI runs in Docker, localhost refers to the container; use host.docker.internal (on Mac/Windows) or your host IP.
docs.openwebui.com

Security: Tools are code. Only install trusted Tools / expose safe OpenAPI endpoints.
docs.openwebui.com
+1

llama.cpp tool use: The server exposes OpenAI-style chat completions and understands function calling/tool metadata; the client (WebUI) can orchestrate execution.
GitHub

If you tell me which route you prefer (Python Tools inside WebUI vs. external OpenAPI servers), I can sketch a minimal example (e.g., a “get weather” tool) wired up to your llama-server in a couple of steps.

You said:
i have a dgx spark, latest llama cpp running in a docker container w/ gpu access, i have latest open webui running in a docker container, its all working correct. im running gpt-oss-120b which supports reasoning and interleaved reasoning and tool calls, i want to support the tools like in the open ai responses api where the server takes care of calling the tool and getting the response on the server, not the client, how does this work?
ChatGPT said:

Thought for 2m 33s
Got it—you want true server-side tool execution (like OpenAI’s Responses API), not browser-side hacks. With your stack (llama.cpp server in Docker + Open WebUI in Docker + a tool-calling model like gpt-oss-120b), there are two viable patterns:

Option A — Use Open WebUI as the server-side tool runner (recommended)
What happens:

Open WebUI sends tools and tool_choice:auto to your llama.cpp server.

The model returns tool_calls.

Open WebUI (server) executes the tool(s) (Python Tools or OpenAPI/MCP tool servers), feeds the tool outputs back to the model, and streams the final message to the client. No client-side execution.
docs.openwebui.com
+3
docs.openwebui.com
+3
docs.openwebui.com
+3

How to wire it:

Start llama.cpp with function calling enabled
Start llama-server with Jinja templates so the server uses the function-calling chat template:

./llama-server --model /models/gpt-oss-120b.gguf --host 0.0.0.0 --port 10000 --jinja
(Function calling is supported when the server runs with --jinja; llama.cpp exposes an OpenAI-compatible /v1/chat/completions.)
GitHub
+1

Point Open WebUI at llama.cpp
Admin → Connections → OpenAI Connections → Add: http://llama-server:10000/v1 (or host.docker.internal/compose service name).
docs.openwebui.com

Register your tools on the server (Open WebUI)

Python Tools: Admin/User → Tools → Add/Install.

OpenAPI/MCP tools: Admin → Tools → + Tool Server, paste your OpenAPI URL (or MCP via the mcpo proxy). Use Global Tool Server so execution runs in the WebUI backend.
docs.openwebui.com
+2
docs.openwebui.com
+2

Enable native function calling for your model
Workspace → Models → (your llama.cpp model) → Function Calling = Native → enable the tools by default. In chat, the + Tools button lets you toggle per-session. (Native = proper tool_calls; Default = prompt-based.)
docs.openwebui.com

That’s it: the model emits tool_calls, Open WebUI executes the tools server-side, then continues the turn—exactly the “server-handled tools” behavior you’re after.
docs.openwebui.com

Option B — Let llama.cpp itself execute a limited set of built-in tools
llama.cpp supports native tool-call formats and mentions built-in tool names such as wolfram_alpha, web_search/brave_search, and code_interpreter. This is narrower than Option A (you don’t get arbitrary OpenAPI tools), but it’s the only thing close to “the model server runs the tool” without WebUI orchestrating. You still send OpenAI-style tools and tool_choice:auto via /v1/chat/completions.
GitHub

Notes

Make sure you start the server with --jinja. You can also pass parallel_tool_calls: true in the request body if you want the model to issue multiple calls at once.
GitHub

Tool-calling quality can drop with aggressive KV quantization; avoid extreme settings like -ctk q4_0 if you see degraded behavior.
GitHub

Minimal request shape (works for both options):

POST /v1/chat/completions
{
  "model": "gpt-oss-120b",
  "messages": [
    {"role":"user","content":"Find the latest on DGX Spark specs and summarize."}
  ],
  "tools": [
    {
      "type":"function",
      "function":{
        "name":"brave_search",
        "description":"Web search",
        "parameters":{
          "type":"object",
          "properties":{"query":{"type":"string"}},
          "required":["query"]
        }
      }
    }
  ],
  "tool_choice":"auto",
  "parallel_tool_calls": true
}
With Option A, Open WebUI executes the tools (including any custom OpenAPI/MCP tools). With Option B, you’re limited to llama.cpp’s built-ins.
docs.openwebui.com
+2
docs.openwebui.com
+2

gpt-oss specifics & gotchas
llama.cpp has an explicit “Hot topics guide: running gpt-oss with llama.cpp” in the README—worth a skim to ensure you’re on the recommended chat template and parameters.
GitHub

Some “tool-capable” OSS models still hesitate to emit tool_calls unless the prompt/system template is right. If calls don’t trigger, switch Open WebUI back to Default (prompt-based) or try another template; then, once it’s reliable, go back to Native.
docs.openwebui.com

TL;DR
If you want Responses-API-style orchestration, use Open WebUI’s server-side tool runner with Function Calling = Native. This executes tools on the server, not the client.
docs.openwebui.com
+1

llama.cpp can recognize some built-in tool names, but for arbitrary tools you still want Open WebUI’s Tool Servers (OpenAPI/MCP) or your own proxy.
GitHub

If you want, tell me which tools you need (HTTP APIs, DB access, filesystem, etc.), and I’ll sketch the exact Open WebUI Tool Server config and a one-file Python Tool so you can flip this on immediately.

You said:
can we do parallel and interleaved tool calls with open webui doing the tool running or do we need to use llama server's hooks to do it?
ChatGPT said:

Thought for 1m 51s
Short answer: you don’t need llama-server “hooks.” Open WebUI can run tools server-side in Native function-calling mode, and it will handle multi-tool chains and “interleaved” (streaming) tool calls just fine as long as streaming is ON. For parallel tool calls:

The model (e.g., your gpt-oss-120b on llama.cpp) decides to emit multiple tool_calls in one turn.

llama.cpp supports this when you pass "parallel_tool_calls": true in the request body.
GitHub

Open WebUI will execute those tool calls server-side, but today they’re typically executed one-by-one, not truly concurrently. Chained multi-tool runs work; full parallel execution is limited.
GitHub
+2
GitHub
+2

What to enable
Function Calling = Native for your model in Open WebUI (Admin → Workspace → Models → Advanced Params). This delegates tool selection to the model and has WebUI execute tool calls server-side.
docs.openwebui.com

Keep streaming ON. Native tool calling relies on streaming so the model can emit tool calls mid-generation and then incorporate results (“interleaving”). Turning streaming off commonly breaks tool loops.
GitHub
+1

Ask llama.cpp for parallel tool calls. llama.cpp accepts "parallel_tool_calls": true in the OpenAI-style request. In WebUI you can pass extra body parameters via its OpenAI-compatible endpoint (or via model “advanced/custom params” depending on your version).
GitHub
+2
GitHub
+2

Example (sent to Open WebUI’s API; WebUI forwards to your llama.cpp endpoint):

POST /api/chat/completions
{
  "model": "gpt-oss-120b",
  "messages": [{"role":"user","content":"Do A and B; use the tools."}],
  "tools": [
    {"type":"function","function":{"name":"toolA","parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}}},
    {"type":"function","function":{"name":"toolB","parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}}
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true
}
llama.cpp may return two tool_calls in one assistant message; Open WebUI executes them server-side and feeds results back to the model in the same turn (interleaved), but execution is generally serial.
GitHub
+1

Practical caveats (so you’re not surprised)
Multiple tool calls do work in WebUI Native mode with capable models, but there have been regressions/edge cases across versions/backends (e.g., confusion with multi-tool calls, UI not always showing large tool results). If something seems off, test with a known-good provider (e.g., GPT-4.5) to isolate a model vs. WebUI issue.
GitHub
+1

Streaming must stay enabled for native tool calling. Disabling streaming is a common cause of “tool never runs” or “doesn’t resume” behavior.
GitHub
+1

WebUI currently has a safety limit on repeated tool loops (e.g., a retry/loop cap) to avoid infinite runs. Don’t rely on arbitrarily deep tool chains without a custom orchestrator.
GitHub

If you truly need concurrent execution
For real parallel I/O (run N tools at once to cut wall-time), wrap them in one orchestrator tool (OpenAPI or MCP) that fans out asynchronously and returns a merged result. From the model’s perspective it’s a single tool call; the server code can use asyncio/threads to parallelize. This sidesteps WebUI’s per-turn serial execution while keeping everything server-side. (WebUI fully supports external tool servers.)
docs.openwebui.com

Bottom line:

Interleaved tool use = ✅ with Open WebUI (Native mode + streaming).
GitHub

Parallel tool selection = ✅ via llama.cpp with "parallel_tool_calls": true.
GitHub

Parallel tool execution inside WebUI = mostly sequential today; use an orchestrator tool if you need real concurrency.
GitHub
+1

If you want, I can sketch a tiny FastAPI “fan-out” tool server that runs N sub-requests concurrently and plugs straight into Open WebUI’s Tool Server UI.

You said:
i have a web search, weather, and news (3 different servers) that expose a couple mcp tools each, how should i set those up to work in open webui as tools to be called by the model server side, the 3 services are running in their own separate docker containers
ChatGPT said:

Thought for 1m 32s
Yes—you can keep all tool execution server-side with Open WebUI. You’ve got two clean ways to register your three MCP services (web search, weather, news):

Option 1: Register the MCP servers directly (if they speak MCP Streamable HTTP)
In Open WebUI: ⚙️ Admin Settings → Features → External Tools → + Add Server

Type: MCP (Streamable HTTP)

URL: http://<container-name-or-ip>:<port>/<mcp-base> (whatever your MCP HTTP endpoint is)

Add auth (OAuth 2.1 / headers) if required.
Save, then enable the tools per chat via the ➕ button (global tools are hidden until toggled).
docs.openwebui.com

Why this works: Open WebUI v0.6.31+ has native MCP support (Streamable HTTP). Requests originate from the WebUI backend (your Docker container), so everything runs server-side.
docs.openwebui.com

Notes
• If your MCP servers use stdio or SSE, they won’t connect directly—use Option 2 (mcpo) below.
docs.openwebui.com

• “Global Tool Servers” are added by admins and don’t auto-appear; toggle them on per chat via the ➕ menu.
docs.openwebui.com

Option 2: Bridge MCP → OpenAPI with mcpo, then add each as an OpenAPI Tool Server (works with stdio/SSE or mixed MCPs)
A. Run one mcpo that fronts all three MCP services
Create an mcpo config that defines your three servers (stdio/SSE/Streamable HTTP all supported):

// config.json
{
  "mcpServers": {
    "websearch": { "type": "streamable-http", "url": "http://websearch:8080/mcp" },
    "weather":   { "type": "sse",             "url": "http://weather:8081/sse" },
    "news":      { "command": "uvx", "args": ["mcp-server-news"] }
  }
}
Run mcpo in Docker on the same network as Open WebUI and your tools:

docker run --name mcpo --restart unless-stopped \
  --network <your-docker-network> -p 8000:8000 \
  -v $(pwd)/config.json:/app/config/config.json \
  ghcr.io/open-webui/mcpo:main \
  --config /app/config/config.json --hot-reload \
  --api-key "top-secret" --root-path "/tools"
This exposes each MCP server under a subpath:
http://mcpo:8000/tools/websearch, .../weather, .../news.
GitHub

B. Register in Open WebUI as OpenAPI Tool Servers
Add each subpath as a Tool Server (don’t use the root):

http://mcpo:8000/tools/websearch

http://mcpo:8000/tools/weather

http://mcpo:8000/tools/news

Include your auth header (e.g., Authorization: Bearer top-secret). Global Tool Servers run from the backend and are toggled on per chat via ➕.
docs.openwebui.com

Why mcpo? It converts any MCP server (stdio/SSE/Streamable HTTP) into clean OpenAPI endpoints that Open WebUI can call server-side. It also auto-docs endpoints at /docs.
docs.openwebui.com
+1

Model-side settings (so llama.cpp + gpt-oss-120b will actually call your tools)
In Open WebUI: Workspace → Models → (your llama.cpp model)

Function Calling: set to Native

Streaming: ON (native tool calling can fail to execute when streaming is off)

(Optional) Add custom param "parallel_tool_calls": true if you want the model to return multiple tool calls in one turn; WebUI will execute them server-side (generally sequentially).
docs.openwebui.com
+1

Docker networking tips (common gotchas)
Use the Docker network service names (e.g., websearch, weather, news, mcpo, openwebui) instead of localhost. Global Tool Servers make requests from the WebUI container, not your browser.
docs.openwebui.com

After adding a Global Tool Server, enable it per chat via ➕ (they’re hidden by default).
docs.openwebui.com

Quick checklist (for your three services)
If they already speak MCP Streamable HTTP → add each as MCP (Streamable HTTP) servers in Open WebUI.
docs.openwebui.com

If they’re stdio/SSE (or you want OpenAPI standardization) → run one mcpo with a config for all three; add each subpath as a separate OpenAPI Tool Server.
docs.openwebui.com
+1

On your model in Open WebUI: Function Calling = Native, Streaming ON (and optionally "parallel_tool_calls": true).
docs.openwebui.com
+1

If you want, tell me the exact container names/ports you’re using and I’ll sketch a docker-compose.yml snippet and the precise URLs to paste into Open WebUI.

You said:
whats sse vs. streamable http and how can i tell what my services are doing. if its worth it, id happily implement whatever is necessary to avoid the proxy
ChatGPT said:

Thought for 1m 39s
Great question. In MCP-land, “SSE” and “Streamable HTTP” are transports—how the client and your tool server exchange JSON-RPC messages.

What’s the difference (in practice)?
Old “HTTP+SSE” (deprecated): two endpoints:
POST to send messages, and a separate GET SSE stream to receive events.
Model Context Protocol

New “Streamable HTTP”: one MCP endpoint (e.g. /mcp) that accepts POST for messages. The server can reply either as a single application/json body or as a text/event-stream (SSE) stream on that same POST response; optionally, a GET to the same endpoint can open a server-initiated SSE stream. This replaces the old HTTP+SSE transport.
Model Context Protocol

Open WebUI’s native MCP integration supports Streamable HTTP only (not stdio or legacy SSE-only servers). So, if you want to avoid any proxy, implement Streamable HTTP on your three services.
docs.openwebui.com

How to tell what your services are doing
Run these quick probes against each service’s MCP endpoint (replace host/port/path):

POST probe (does it speak Streamable HTTP?)

curl -i -X POST http://toolhost:8000/mcp \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  --data '[{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"clientInfo":{"name":"probe","version":"0.1"}}}]'
If you get Content-Type: application/json with a JSON-RPC result (or text/event-stream streaming responses) → it’s Streamable HTTP.

If you get 4xx/405 here but a GET (below) works, it may be an older HTTP+SSE server.
Model Context Protocol

GET probe (does it expose a server-push stream?)

curl -i -N -H 'Accept: text/event-stream' http://toolhost:8000/mcp
200 + Content-Type: text/event-stream → it offers a server-initiated SSE stream (optional under Streamable HTTP).

405 Method Not Allowed → that’s fine; many Streamable HTTP servers don’t expose a GET stream.

If it’s the old HTTP+SSE transport, first SSE event is often an endpoint metadata event per the back-compat guidance.
Model Context Protocol

Headers tell-tales

Streamable HTTP servers may return Mcp-Session-Id on initialize; if present, you must echo it on subsequent requests.
Model Context Protocol

Code/config hints

Python servers using FastMCP or the MCP Python SDK will have mcp.run(transport="streamable-http", ...).

TypeScript servers using the official SDK will create/connect a Streamable HTTP transport instead of StdioServerTransport.
jlowin.dev
+1

If you want to avoid the proxy (mcpo): make each service Streamable HTTP
Python (FastMCP) minimal
from fastmcp import FastMCP
mcp = FastMCP("weather")

@mcp.tool()
def forecast(city: str) -> str:
    # ... your logic ...
    return "Sunny"

mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, path="/mcp")
This exposes a single /mcp endpoint that speaks Streamable HTTP and works directly with Open WebUI.
jlowin.dev
+1

TypeScript (official SDK) sketch
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHttpServerTransport } from "@modelcontextprotocol/sdk/server/streamable-http.js";

const server = new McpServer({ name: "news", version: "1.0.0" });
server.tool("top_headlines", { q: z.string().optional() }, async ({ q }) => {
  // ... your logic ...
  return { content: [{ type: "text", text: "..." }] };
});

const transport = new StreamableHttpServerTransport({ host: "0.0.0.0", port: 8002, path: "/mcp" });
await server.connect(transport);
(The SDK includes examples for Streamable HTTP and back-compat servers.)
github.com
+1

Wire into Open WebUI
In Admin → Features → Model Context Protocol (MCP) → Add Server, choose Type: MCP (Streamable HTTP) and paste each URL, e.g.:

http://websearch:8000/mcp

http://weather:8001/mcp

http://news:8002/mcp
Toggle them on in chat. Open WebUI doesn’t support stdio/legacy SSE natively, hence the need for Streamable HTTP to avoid a proxy.
docs.openwebui.com

Why Streamable HTTP is the right target here
It replaces the older HTTP+SSE transport and unifies request/stream semantics on one endpoint.
Model Context Protocol

It’s the only MCP transport Open WebUI supports natively.
docs.openwebui.com

If you share one of your service endpoints, I can run through the curl probes with you and point out exactly which transport it’s speaking and what to change.

You said:
here is an example mcp server i would want to convert, as an example: """
Code Interpreter MCP Server

Provides sandboxed Python code execution via Docker containers.
Enforces security constraints: network disabled, resource limits, timeout.
Supports matplotlib figure capture and serialization.
"""

import base64
import docker
import io
import json
import logging
import sys
from mcp.server.fastmcp import FastMCP
from tool_progress import ToolProgress, NullProgress

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("code-interpreter")

# Initialize Docker client
try:
    docker_client = docker.from_env()
    logger.info("Docker client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Docker client: {e}")
    docker_client = None


def _should_wrap_matplotlib(code: str) -> bool:
    """Check if code actually uses matplotlib or seaborn (not just imports it).

    Args:
        code: User's Python code

    Returns:
        True if matplotlib or seaborn is actually used in the code
    """
    # Check for actual matplotlib and seaborn usage patterns, not just imports
    # Look for plt./sns. or pyplot/seaborn usage, which indicates real visualization code
    visualization_usage_patterns = [
        'plt.',              # matplotlib.pyplot alias
        'pyplot.',           # Direct pyplot module usage
        'matplotlib.pyplot', # Full module path
        'sns.',              # seaborn alias
        'seaborn.',          # seaborn module
        'plt.show()',        # Explicit show call (even though we don't need it)
        'plt.plot(',         # matplotlib plotting functions
        'plt.scatter(',
        'plt.bar(',
        'plt.hist(',
        'plt.imshow(',
        'sns.histplot(',     # seaborn plotting functions
        'sns.scatterplot(',
        'sns.lineplot(',
        'sns.barplot(',
        'sns.boxplot(',
        'sns.violinplot(',
        'sns.heatmap(',
        'sns.pairplot(',
    ]

    return any(pattern in code for pattern in visualization_usage_patterns)


def _wrap_code_for_figure_capture(code: str) -> str:
    """Wrap user code to capture matplotlib/seaborn figures if visualization code is used.

    Only adds wrapper if code actually uses matplotlib or seaborn.
    Always returns code that uses our figure capture mechanism.

    Args:
        code: User's Python code

    Returns:
        Wrapped code with figure capture (if matplotlib/seaborn is used)
    """
    if not _should_wrap_matplotlib(code):
        # Code doesn't use matplotlib, execute as-is
        return code

    # Code uses matplotlib, wrap it with capture code
    # Set up matplotlib with Agg backend and capture code at the end
    wrapper_start = '''import sys
import io
import base64
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

_CAPTURED_FIGURES = []

def _capture_figures():
    """Capture all matplotlib figures as base64 PNG strings."""
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            _CAPTURED_FIGURES.append(img_base64)
        except Exception as e:
            print(f"Error capturing figure: {e}", file=sys.stderr)
    plt.close('all')
    return _CAPTURED_FIGURES

# User code starts here
'''

    wrapper_end = '''
# Figure capture happens here
_figures = _capture_figures()
if _figures:
    print("__EXECUTION_RESULT__" + json.dumps({"images": _figures}))
'''

    return wrapper_start + code + wrapper_end


@mcp.tool()
def execute_python(code: str, tool_call_id: str = None) -> str:
    """Execute Python code in a sandboxed Docker container.

    Security features:
    - Runs as non-root user
    - Network disabled
    - Resource limits: 1GB RAM, 1 CPU
    - Container removed after execution

    Visualization Support:
    - Matplotlib and seaborn figures are AUTOMATICALLY CAPTURED and DISPLAYED
    - Any figures you create via plt.plot(), sns.histplot(), etc. will be automatically
      shown to the user - you don't need to do anything special
    - Do NOT try to manually capture or encode figures (the system handles this)

    Returns unified JSON format:
    {
        "stdout": "text output",
        "stderr": "error output if any",
        "images": ["base64_png_1", "base64_png_2"],  # Internal use only - figures displayed separately
        "success": true/false
    }

    Args:
        code: Python code to execute (string)
        tool_call_id: Internal parameter for progress tracking (injected by backend)

    Returns:
        JSON string with stdout, stderr, images, and success flag

    Example:
        execute_python("print('Hello, world!')")
        execute_python("import numpy as np; print(np.array([1,2,3]).sum())")
        execute_python("import matplotlib.pyplot as plt\\nplt.plot([1,2,3])\\nplt.title('My Plot')")
        execute_python("import seaborn as sns\\ndata = [1,2,2,3,3,3]\\nsns.histplot(data)")
    """
    # Create progress tracker
    progress = ToolProgress(tool_call_id) if tool_call_id else NullProgress()

    if docker_client is None:
        progress.error("Docker client not available")
        return json.dumps({
            "stdout": "",
            "stderr": "Docker client not available. Is Docker running?",
            "images": [],
            "success": False
        })

    try:
        logger.info(f"Executing Python code (length: {len(code)} chars)")
        progress.update("preparing", "Preparing Docker container...")

        # Wrap code to capture matplotlib figures (only if matplotlib is used)
        wrapped_code = _wrap_code_for_figure_capture(code)
        progress.update("executing", f"Executing {len(code)} chars of Python code...")

        result = docker_client.containers.run(
            "code-executor:latest",
            command=["python3", "-c", wrapped_code],
            remove=True,  # Remove container after execution
            mem_limit="1g",  # 1GB RAM limit
            cpu_period=100000,
            cpu_quota=100000,  # 1 CPU limit
            network_disabled=True,  # No network access
            stdout=True,
            stderr=True,
            user="executor"  # Run as non-root user
        )

        output = result.decode('utf-8')
        logger.info(f"Code execution successful (output length: {len(output)} chars)")
        progress.complete("Executed successfully")

        # Parse output for execution result and matplotlib figures
        stdout = output
        images = []

        if "__EXECUTION_RESULT__" in output:
            # Split output into stdout and execution result
            parts = output.split("__EXECUTION_RESULT__", 1)
            stdout = parts[0].strip()

            # Parse execution result JSON (contains images)
            try:
                result_json = json.loads(parts[1])
                images = result_json.get("images", [])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse execution result: {e}")

        # Return unified format
        result_obj = {
            "stdout": stdout,
            "stderr": "",
            "images": images,
            "success": True
        }
        return json.dumps(result_obj)

    except docker.errors.ContainerError as e:
        # Container exited with non-zero code
        stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
        stdout = e.stdout.decode('utf-8') if e.stdout else ""

        logger.warning(f"Container execution error: {stderr}")
        progress.error(f"Execution error: {stderr[:100]}")

        result_obj = {
            "stdout": stdout,
            "stderr": stderr,
            "images": [],
            "success": False
        }
        return json.dumps(result_obj)

    except docker.errors.ImageNotFound:
        logger.error("code-executor:latest image not found")
        error_msg = "code-executor Docker image not found. Please build it first: docker build -t code-executor:latest ./code-executor"
        progress.error("Docker image not found")

        result_obj = {
            "stdout": "",
            "stderr": error_msg,
            "images": [],
            "success": False
        }
        return json.dumps(result_obj)

    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        error_msg = f"Docker API Error: {str(e)}"
        progress.error(error_msg)

        result_obj = {
            "stdout": "",
            "stderr": error_msg,
            "images": [],
            "success": False
        }
        return json.dumps(result_obj)

    except Exception as e:
        logger.error(f"Unexpected error during code execution: {e}", exc_info=True)
        error_msg = f"Unexpected Error: {str(e)}"
        progress.error(error_msg)

        result_obj = {
            "stdout": "",
            "stderr": error_msg,
            "images": [],
            "success": False
        }
        return json.dumps(result_obj)


@mcp.tool()
def get_available_libraries() -> str:
    """Get list of pre-installed Python libraries in the code executor.

    Returns:
        List of available libraries (formatted as string)
    """
    libraries = [
        "numpy - Numerical computing",
        "pandas - Data analysis",
        "matplotlib - Plotting",
        "requests - HTTP client",
        "scipy - Scientific computing",
        "sympy - Symbolic mathematics"
    ]

    return "Available Python libraries:\n" + "\n".join(f"  • {lib}" for lib in libraries)


if __name__ == "__main__":
    logger.info("Starting Code Interpreter MCP server...")
    mcp.run(transport="stdio")
ChatGPT said:

Thought for 1m 4s
Awesome—this server is already 95% there. To make it work directly with Open WebUI (no proxy), you just need to switch the transport to Streamable HTTP and (optionally) return structured MCP content. Here’s exactly what to change and how to wire it up.

1) Convert to Streamable HTTP (drop-in change)
What changes:

Keep your tools and logic exactly as-is.

Run the server with transport="streamable-http" on a single endpoint (e.g., /mcp).

Put it on the same Docker network as Open WebUI and llama.cpp.

Minimal patch (only the parts that change):

# + add:
import os

# ... keep the rest of your file unchanged ...

if __name__ == "__main__":
    logger.info("Starting Code Interpreter MCP server (Streamable HTTP)...")
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        path="/mcp"
    )
Notes

Your existing @mcp.tool() functions and type hints are fine—FastMCP will expose them automatically over MCP.

If you currently from mcp.server.fastmcp import FastMCP, that’s OK. If you use the standalone package, the import is from fastmcp import FastMCP. Either works as long as your runtime has the right package.

2) (Optional but nice) Return structured MCP content
Right now you return json.dumps({...}) strings. That works, but you can give Open WebUI richer data (including images) by returning MCP content parts. For example:

@mcp.tool()
def execute_python(code: str, tool_call_id: str = None):
    # ... run the container, build result_obj as you already do ...
    # result_obj = {"stdout": "...", "stderr": "...", "images": ["b64...", ...], "success": True}

    parts = []
    if result_obj["stdout"]:
        parts.append({"type": "text", "text": result_obj["stdout"]})
    if result_obj["stderr"]:
        parts.append({"type": "text", "text": f"[stderr]\\n{result_obj['stderr']}"})
    for b64 in result_obj["images"]:
        parts.append({"type": "image", "data": b64, "mimeType": "image/png"})

    # You can also keep a compact JSON blob for programmatic use:
    parts.append({"type": "text", "text": json.dumps({
        "success": result_obj["success"],
        "n_images": len(result_obj["images"])
    })})

    return {"content": parts}
If you want to keep it simple, returning a plain string is fine; this is just a nicer UX (images render inline in many clients).

3) Verify your service speaks Streamable HTTP
POST (primary path)

curl -i -X POST http://localhost:8000/mcp \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  --data '[{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"clientInfo":{"name":"probe","version":"0.1"}}}]'
Expect 200 with Content-Type: application/json (or text/event-stream if it streams the response).

(Optional) GET stream

curl -i -N -H 'Accept: text/event-stream' http://localhost:8000/mcp
Some servers expose a GET SSE feed too (not required). 405 here is fine.

4) Docker wiring
docker-compose.yml (excerpt)

version: "3.9"
networks:
  ai:
    driver: bridge

services:
  openwebui:
    image: ghcr.io/open-webui/open-webui:latest
    container_name: openwebui
    ports: ["3000:8080"]
    networks: [ai]
    environment:
      - OAUTH_DISABLE=True   # if you’re testing locally

  llama:
    image: ghcr.io/ggerganov/llama.cpp:server
    container_name: llama
    command: ["--model", "/models/gpt-oss-120b.gguf", "--host", "0.0.0.0", "--port", "10000", "--jinja"]
    ports: ["10000:10000"]
    volumes: ["/path/to/models:/models:ro"]
    networks: [ai]

  code_interpreter:
    build: ./code-interpreter
    container_name: code-interpreter
    environment:
      - PORT=8000
    ports: ["8000:8000"]
    networks: [ai]
    # You need Docker access to spawn the sandbox container:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    # (security: consider scoping this host, cgroup constraints, and a dedicated Docker socket if possible)
5) Add it to Open WebUI (no proxy)
Open WebUI → Admin Settings → Features → Model Context Protocol (MCP) → Add Server

Type: MCP (Streamable HTTP)

URL: http://code-interpreter:8000/mcp

Save. Then in a chat, hit ➕ and toggle the code-interpreter tools on.

Ensure your llama.cpp-backed model has Function Calling = Native and Streaming ON in Open WebUI’s model settings.

Repeat those steps for your web search, weather, and news servers (each with their own container + /mcp endpoint). No mcpo or OpenAPI bridge needed.

6) A couple of quality-of-life tips
Parallel tool selection: If you want the model to emit multiple tool calls at once, pass "parallel_tool_calls": true via Open WebUI’s advanced model params. Open WebUI will still execute them server-side; execution is typically sequential—if you want real concurrency, put the fan-out inside the tool itself.

Progress updates: If you want the UI to show “preparing/executing/completed,” you can keep your ToolProgress and also consider emitting incremental text/event-stream chunks on long runs; FastMCP’s Streamable HTTP transport will carry them.

Images: You’re already base64-encoding; returning {type:"image", data:"...", mimeType:"image/png"} content parts lets clients render plots inline.

If you paste your requirements.txt / Dockerfile for this service, I’ll give you a production-safe build (small base image, pinned deps, non-root, and a healthcheck) that you can clone for the other two MCP services.

You said:
The tool progress was custom built for a different client. How do we adopt MCP more fully with stream able http in the most standard way (for all of our more sophisticated MCP functionality actually
ChatGPT said:

Thought for 3m 37s
Great question — you can ditch your bespoke ToolProgress and go fully “MCP-native” with Streamable HTTP. Here’s the pragmatic migration plan and a minimal patch you can apply across your servers (including the Docker-sandboxed code interpreter you shared).

What to change (high-level)
Run your servers over Streamable HTTP.
In FastMCP/Python this is literally one flag or an ASGI mount.

# simplest: one-process server
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
Or mount into an existing Starlette/Uvicorn app via streamable_http_app() if you prefer (handy when you have several servers).
GitHub

Replace custom progress with MCP progress notifications.
Clients ask for progress by sending a _meta.progressToken in their request; your server emits notifications/progress during the call. FastMCP gives you ctx.report_progress(...) so you don’t have to craft JSON-RPC yourself.
Model Context Protocol
+1

Use MCP logging notifications instead of ad-hoc logs.
Send structured logs via ctx.debug/info/warning/error. Clients enable them with logging/setLevel.
Model Context Protocol

Implement cancellation.
MCP clients can send notifications/cancelled for an in-flight request; your tool should stop work and clean up. (You’ll typically see the SDK propagate this so your long task can break or be cancelled.)
Model Context Protocol

Return normal tool results; let the SDK shape them.
FastMCP maps Python return values to MCP “content.” You can return dicts/Pydantic models (for structured output) and Image objects for images (instead of manual base64 bundling).
GitHub

Mind Streamable HTTP session headers + CORS.
If the server returns a sessionId on initialize, clients must send it back in the Mcp-Session-Id header on subsequent requests. Expose it via CORS if the client is browser-based.
Model Context Protocol

Minimal patch to your example server
Below is a tight edit of your code interpreter that:

swaps ToolProgress for MCP progress/logging,

supports Streamable HTTP,

uses typed images (optional but cleaner),

is ready for Open WebUI (which talks Streamable HTTP).

"""
Code Interpreter MCP Server (Streamable HTTP, MCP-native progress/logging)
"""

import base64, io, json, logging, docker
from mcp.server.fastmcp import FastMCP, Context, Image  # Image is optional but nice
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mcp = FastMCP("code-interpreter")
docker_client = docker.from_env()

def _should_wrap_matplotlib(code: str) -> bool:
    patterns = ['plt.', 'pyplot.', 'matplotlib.pyplot', 'sns.', 'seaborn.',
                'plt.show()', 'plt.plot(', 'plt.scatter(', 'plt.bar(', 'plt.hist(',
                'plt.imshow(', 'sns.histplot(', 'sns.scatterplot(', 'sns.lineplot(',
                'sns.barplot(', 'sns.boxplot(', 'sns.violinplot(', 'sns.heatmap(',
                'sns.pairplot(']
    return any(p in code for p in patterns)

def _wrap_code_for_figure_capture(code: str) -> str:
    if not _should_wrap_matplotlib(code):
        return code
    return r'''import sys, io, base64, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_CAPT=[]
def _cap():
    figs=[plt.figure(n) for n in plt.get_fignums()]
    for f in figs:
        try:
            b=io.BytesIO(); f.savefig(b, format="png", dpi=100, bbox_inches="tight")
            b.seek(0); _CAPT.append(base64.b64encode(b.read()).decode("utf-8"))
        except Exception as e:
            print("Error capturing figure:", e, file=sys.stderr)
    plt.close("all"); return _CAPT
# user code:
''' + code + r'''
# capture:
print("__EXECUTION_RESULT__"+json.dumps({"images": _cap()}))
'''

@mcp.tool()
async def get_available_libraries() -> list[str]:
    return [
        "numpy - Numerical computing",
        "pandas - Data analysis",
        "matplotlib - Plotting",
        "requests - HTTP client",
        "scipy - Scientific computing",
        "sympy - Symbolic mathematics",
    ]

@mcp.tool()
async def execute_python(code: str, ctx: Context) -> dict:
    """
    Execute Python code in a sandboxed Docker container.

    Emits MCP-native progress/logging; returns structured result.
    """
    await ctx.info(f"Executing Python ({len(code)} chars)")
    await ctx.report_progress(0.05, 1.0, "Preparing container")  # MCP progress

    try:
        wrapped = _wrap_code_for_figure_capture(code)
        await ctx.report_progress(0.25, 1.0, "Starting container")

        result = docker_client.containers.run(
            "code-executor:latest",
            command=["python3", "-c", wrapped],
            remove=True,
            mem_limit="1g",
            cpu_period=100_000, cpu_quota=100_000,
            network_disabled=True,
            stdout=True, stderr=True,
            user="executor",
        )
        await ctx.report_progress(0.80, 1.0, "Processing output")

        output = result.decode("utf-8")
        stdout, images = output, []
        if "__EXECUTION_RESULT__" in output:
            head, tail = output.split("__EXECUTION_RESULT__", 1)
            stdout = head.strip()
            try:
                images_b64 = json.loads(tail).get("images", [])
                # Option A (MCP-native): return Image objects
                images = [Image(data=base64.b64decode(b), format="png") for b in images_b64]
            except json.JSONDecodeError:
                await ctx.warning("Failed to parse figure JSON")

        await ctx.report_progress(1.0, 1.0, "Done")

        # Return a structured dict; SDK converts to MCP content
        return {
            "stdout": stdout,
            "stderr": "",
            "images_count": len(images),
            "success": True,
            "images": images,  # Image objects or omit if you prefer your old base64
        }

    except docker.errors.ImageNotFound:
        msg = ("code-executor:latest not found. Build it: "
               "docker build -t code-executor:latest ./code-executor")
        await ctx.error(msg)
        return {"stdout": "", "stderr": msg, "images_count": 0, "success": False}

    except docker.errors.ContainerError as e:
        stderr = e.stderr.decode("utf-8") if e.stderr else str(e)
        stdout = e.stdout.decode("utf-8") if e.stdout else ""
        await ctx.warning(f"Container error: {stderr[:120]}")
        return {"stdout": stdout, "stderr": stderr, "images_count": 0, "success": False}

    except Exception as e:
        await ctx.error(f"Unexpected error: {e!r}")
        return {"stdout": "", "stderr": str(e), "images_count": 0, "success": False}

if __name__ == "__main__":
    # Run with Streamable HTTP (Open WebUI talks this)
    mcp.run(transport="streamable-http")
Notes on the patch

ctx.report_progress(...) emits MCP notifications/progress behind the scenes when the client supplied a _meta.progressToken in the original request. You don’t have to manage tokens yourself.
Model Context Protocol
+1

ctx.info/debug/warning/error(...) emit MCP logging notifications (notifications/message) once the client has set a log level via logging/setLevel.
Model Context Protocol

Image is an SDK helper; the SDK serializes it into proper MCP image content so UIs can render thumbnails inline.
GitHub

Keep your existing Docker limits (non-root, network off). Cancellation will arrive as notifications/cancelled; ensure your long work can be interrupted (e.g., periodically check for cancellation or make container runs time-bounded).
Model Context Protocol

Deployment + interop tips
Multiple servers, one port. Mount each MCP server on its own path with Starlette and Mount(...) (e.g., /code, /weather, /news). The SDK shows ready-made examples.
GitHub
+1

Open WebUI integration. Point Open WebUI at each MCP Streamable HTTP endpoint (it advertises first-class Streamable HTTP support). No extra proxy needed.

Sessions & CORS. If your server returns a sessionId on initialize, browser clients must echo it via the Mcp-Session-Id header. Configure CORS to expose/allow that header.
Model Context Protocol

Why this is “the standard way”
Transports: Streamable HTTP is the spec’d remote transport; it keeps a bidirectional stream for interleaved progress/logs + final results.
Model Context Protocol

Tools: Use the protocol’s tools/list / tools/call and let your server own execution (server-side calling and aggregation).
Model Context Protocol

Progress/Logging/Cancellation: All first-class parts of the MCP spec; clients know how to show them without any custom glue.
Model Context Protocol
+2
Model Context Protocol
+2

If you want, I can also sketch a tiny compose file wiring your three service containers (web search, weather, news) behind a Starlette router with /search, /weather, /news MCP paths so Open WebUI sees all three as tools from one base URL.





No file chosenNo file chosen
ChatGPT can make mistakes. OpenAI doesn't use StrieberGPT workspace data to train its models.

