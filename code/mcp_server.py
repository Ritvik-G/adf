"""
MCP Calculator Server
Exposes add, subtract, multiply, divide as MCP tools over streamable HTTP.
Runs on port 8100 to avoid conflict with REST (5001) and SOAP (8000).
Also exposes /metrics so the benchmark can measure server-side CPU/memory
cost separately from client-side cost.
"""

from mcp.server.mcpserver import MCPServer
from starlette.responses import JSONResponse
from monitor import SystemMonitor

mcp = MCPServer("calculator", log_level="WARNING")
monitor = SystemMonitor()
monitor.start()


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers: a + b"""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a: a - b"""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers: a * b"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b: a / b"""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


@mcp.custom_route("/metrics", methods=["GET"])
async def metrics(request):
    # stop() records the end CPU time, which stats() needs for cpu_seconds.
    # The benchmark reads /metrics once at the end of a run, then resets.
    monitor.stop()
    return JSONResponse(monitor.stats())


@mcp.custom_route("/metrics/reset", methods=["POST"])
async def metrics_reset(request):
    monitor.stop()
    monitor.start()
    return JSONResponse({"status": "reset"})


if __name__ == "__main__":
    print("Starting MCP server on http://localhost:8100/mcp")
    mcp.run("streamable-http", port=8100)
