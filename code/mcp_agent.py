"""
MCP Agent
An LLM is given the calculator's MCP tools (add, subtract, multiply, divide)
and must solve the equation by calling them. Works with Groq and Ollama.

The agent loop:
  1. Send the equation and the tool definitions to the LLM.
  2. If the LLM asks for tool calls, run each one on the MCP server and send
     the results back. Repeat.
  3. When the LLM replies without tool calls, that reply is its final answer,
     graded strictly like every other method (must be a clean number).
"""

import json
import requests
from config import load_llm_config

SYSTEM_PROMPT = (
    "You are a calculator agent. Use the provided tools for every arithmetic "
    "operation, one operation per tool call. When you have the final result, "
    "reply with ONLY the final numerical value, no explanations or equations."
)


def tools_for_llm(session):
    """Convert MCP tool definitions to the function-calling format Groq and Ollama both accept"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in session.list_tools()
    ]


# ============================================================================
# LLM backends
#
# Each chat function returns (assistant_message, content, tool_calls, stats):
#   assistant_message — the reply, ready to append to the message history
#   content           — the reply text (the final answer if no tool calls)
#   tool_calls        — list of {"id", "name", "args"}
#   stats             — token usage (Groq) or timing (Ollama) for this turn,
#                       using the same keys benchmark.py already aggregates
# Each tool_message function builds the message that returns a tool result.
# ============================================================================

def chat_groq(messages, tools, temperature):
    cfg = load_llm_config()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
        "max_tokens": cfg.get("max_tokens", 256),
        "top_p": cfg.get("top_p", 1.0),
    }

    r = requests.post(cfg["url"], headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    data = r.json()
    message = data["choices"][0]["message"]

    content = (message.get("content") or "").strip()
    raw_calls = message.get("tool_calls") or []
    tool_calls = [
        {"id": c["id"], "name": c["function"]["name"], "args": json.loads(c["function"]["arguments"])}
        for c in raw_calls
    ]

    assistant_message = {"role": "assistant", "content": content}
    if raw_calls:
        assistant_message["tool_calls"] = raw_calls

    usage = data.get("usage", {})
    stats = {k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return assistant_message, content, tool_calls, stats


def groq_tool_message(call, output):
    return {"role": "tool", "tool_call_id": call["id"], "content": output}


def chat_ollama(messages, tools, model, temperature):
    cfg = load_llm_config()
    base_url = cfg.get("ollama_url", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        # Ollama only reads sampling settings from "options". num_predict caps
        # the reply length (same max_tokens as Groq), so a model stuck
        # repeating itself stops instead of generating until the timeout.
        "options": {"temperature": temperature, "num_predict": cfg.get("max_tokens", 256)},
    }

    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    data = r.json()
    message = data["message"]

    content = (message.get("content") or "").strip()
    raw_calls = message.get("tool_calls") or []
    tool_calls = [
        {"id": None, "name": c["function"]["name"], "args": c["function"]["arguments"]}
        for c in raw_calls
    ]

    assistant_message = {"role": "assistant", "content": content}
    if raw_calls:
        assistant_message["tool_calls"] = raw_calls

    stats = {
        "total_duration_ns": data.get("total_duration", 0),
        "load_duration_ns": data.get("load_duration", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
    }
    return assistant_message, content, tool_calls, stats


def ollama_tool_message(call, output):
    return {"role": "tool", "tool_name": call["name"], "content": output}


# ============================================================================
# Agent loop
# ============================================================================

def run_agent(equation, session, chat, tool_message, tools, max_steps=6):
    """
    Solve one equation with an LLM that calls MCP tools.

    Args:
        equation (str): Mathematical equation to evaluate
        session (MCPSession): Open session from mcp_client.connect()
        chat: Function (messages, tools) -> (assistant_message, content, tool_calls, stats)
        tool_message: Function (call, output) -> message returning a tool result
        tools (list): Tool definitions from tools_for_llm()
        max_steps (int): Maximum LLM turns before giving up

    Returns:
        (str or None, dict): The final answer (None if the LLM call failed
        or max_steps ran out), and stats for this equation: summed token
        usage or timing, plus llm_turns, tool_calls, tool_errors,
        no_tool_answers and max_steps_exceeded.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {equation}\nANSWER:"},
    ]
    stats = {"llm_turns": 0, "tool_calls": 0, "tool_errors": 0, "no_tool_answers": 0, "max_steps_exceeded": 0}

    for _ in range(max_steps):
        try:
            assistant_message, content, tool_calls, turn_stats = chat(messages, tools)
        except Exception as e:
            print(f"  LLM Error: {e}")
            return None, stats

        stats["llm_turns"] += 1
        for k, v in turn_stats.items():
            stats[k] = stats.get(k, 0) + v

        if not tool_calls:
            if stats["tool_calls"] == 0:
                stats["no_tool_answers"] = 1
            return content, stats

        messages.append(assistant_message)
        for call in tool_calls:
            stats["tool_calls"] += 1
            result = session.call_tool(call["name"], call["args"])
            if result.is_error:
                stats["tool_errors"] += 1
            messages.append(tool_message(call, result.content[0].text))

    stats["max_steps_exceeded"] = 1
    return None, stats
