import re
from contextlib import contextmanager
from anyio.from_thread import start_blocking_portal
from mcp.client import Client

MCP_URL = "http://localhost:8100/mcp"

OPERATION_MAP = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
}


class MCPSession:
    """Synchronous handle to an open MCP client session (see connect())."""

    def __init__(self, portal, client):
        self._portal = portal
        self._client = client

    def list_tools(self):
        """Return the server's tool definitions"""
        return self._portal.call(self._client.list_tools).tools

    def call_tool(self, name, arguments):
        """Call a tool and return the raw CallToolResult"""
        return self._portal.call(self._client.call_tool, name, arguments)


@contextmanager
def connect(url=MCP_URL):
    """
    Open one MCP session for the whole run, not one per equation.

    The official mcp client is async-only, while the rest of this project is
    synchronous. anyio's blocking portal runs the async client in a
    background event loop and lets plain synchronous code call into it.

    Usage:
        with connect() as session:
            session.call_tool("add", {"a": 1, "b": 2})
    """
    with start_blocking_portal() as portal:
        with portal.wrap_async_context_manager(Client(url)) as client:
            yield MCPSession(portal, client)


def make_mcp_call(session, operation, a, b, verbose=True):
    """Make an MCP tool call for the given arithmetic operation"""
    response = session.call_tool(operation, {"a": float(a), "b": float(b)})
    if response.is_error:
        raise RuntimeError(response.content[0].text)
    result = float(response.structured_content["result"])
    if verbose:
        print(f"MCP Call: {operation}({a}, {b}) = {result}")
    return result


def evaluate_expression(expression, session, verbose=True):
    """
    Evaluate expression by repeatedly finding and solving innermost operations via MCP

    Args:
        expression (str): Mathematical expression to evaluate
        session (MCPSession): Open session from connect()
        verbose (bool): Print step-by-step execution

    Returns:
        float: Final result of the expression
    """
    expression = expression.replace(" ", "")

    if verbose:
        print(f"Original Expression: {expression}\n")

    # Pattern to match: number operator number
    pattern = r"\(?\s*(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*\)?"

    step = 1
    while True:
        match = re.search(pattern, expression)

        if not match:
            result = float(expression.strip("()"))
            if verbose:
                print(f"\nFinal Result: {result}")
            return result

        a, operator, b = match.groups()
        operation = OPERATION_MAP[operator]

        if verbose:
            print(f"Step {step}:")

        result = make_mcp_call(session, operation, a, b, verbose=verbose)

        expression = expression[: match.start()] + str(result) + expression[match.end() :]

        if verbose:
            print(f"Expression now: {expression}\n")

        step += 1
