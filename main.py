#!/usr/bin/env python3
"""Function-calling server for llama-server's OpenAI-compatible chat completions API.

This module handles HTTP transport, tool management, and the conversation loop.
It is display-agnostic – all output goes through an `emit` callback provided
by the client (e.g. terminal.py).

Start llama-server first, e.g.:
    ./llama.cpp/build/bin/llama-server --jinja --fim-qwen-30b-default
"""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import os
import signal
import subprocess
import sys
import urllib.request
import aiohttp
from collections.abc import Iterator
from typing import AsyncIterator, Callable, get_type_hints, Protocol, TypedDict
import random
import string
from datetime import datetime

# Load environment variables with priority order
# 1. .env.example (defaults)
# 2. .env (user overrides - if exists)
try:
    from dotenv import load_dotenv
    
    # First load defaults from .env.example
    load_dotenv(".env.example", override=False)
    
    # Then load .env if it exists (overrides .env.example)
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
    
    # Set fallback defaults if neither file has them
    if "LLAMA_BASE_URL" not in os.environ:
        os.environ["LLAMA_BASE_URL"] = "http://127.0.0.1:8080"
except ImportError:
    # If python-dotenv is not installed, use hardcoded defaults
    if "LLAMA_BASE_URL" not in os.environ:
        os.environ["LLAMA_BASE_URL"] = "http://127.0.0.1:8080"
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()



# Import tools from tools.py
from tools import (
    run_bash,
    load_model,
    unload_model,
    list_models,
    load_skills,
    make_tool
)



CURRENT_MODEL = os.environ.get("CURRENT_MODEL", "Qwen_Qwen3.5-27B-Q4_1")

def is_llama_running(base_url: str) -> bool:
    """Check if llama-server is already running by making a request to the API."""
    try:
        import urllib.request
        response = urllib.request.urlopen(f"{base_url}/v1/models", timeout=2)
        return response.status == 200
    except Exception:
        return False


def load_model_and_set(model_name: str):
    global CURRENT_MODEL
    load_model(model_name)
    CURRENT_MODEL = model_name

# Import vector_search and download_model from their modules
from vector_search import vector_search
from download_model import download_model

# ---------------------------------------------------------------------------
# Types – Chat Completions API
# ---------------------------------------------------------------------------

from main_types import * 
from conversation import (
    approximate_token_count,
    archive_conversation,
    write_conversation,
    read_conversation,
    get_last_conversation_session_id,
)

# ---------------------------------------------------------------------------
# Pure constructors
# ---------------------------------------------------------------------------

_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool_from_function(fn: ToolHandler) -> Tool:
    """Build a Tool from a function's signature and docstring."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties = {}
    required = []
    for name, p in sig.parameters.items():
        properties[name] = {"type": _TYPE_MAP.get(hints.get(name, str), "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return make_tool(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        parameters={"type": "object", "properties": properties, "required": required},
    )


def make_user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def make_system_message(content: str) -> Message:
    return {"role": "system", "content": content}


def make_tool_result_message(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = pathlib.Path(__file__).parent
SYSTEM_MD = ROOT_DIR / "SYSTEM.md"
SKILLS_DIR = ROOT_DIR / "skills"


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


async def stream_chat_completion(
    base_url: str,
    messages: list[Message],
    tools: list[Tool], 
    session_id: SessionID
) -> AsyncIterator[ChatCompletionChunk]:
    """Send a streaming chat completion request, yielding SSE chunks asynchronously."""
    system_content = SYSTEM_MD.read_text() if SYSTEM_MD.exists() else ""
    
    # Add dynamic context to system prompt
    session_context = f"\n\nCurrent time: {datetime.now().isoformat()}\nCurrent working directory: {os.getcwd()}\nActive session: {session_id}\n"
    system_content += session_context

    # Strip timestamps before sending to LLM API
    msgs = [make_system_message(system_content)] + [
        {k: v for k, v in m.items() if k != "timestamp"} 
        for m in messages if m.get("role") != "system"
    ]
    body: ChatCompletionRequest = {
        "model": CURRENT_MODEL,
        "messages": msgs,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"}
        ) as resp:
            logger.debug(f"loading: {base_url}/v1/chat/completions")
            async for line in resp.content:
                raw_line = line.decode("utf-8").strip()
                if not raw_line or raw_line.startswith(":"):
                    continue
                if raw_line == "data: [DONE]":
                    break
                if raw_line.startswith("data: "):
                    yield json.loads(raw_line[6:])


async def consume_stream(
    chunks: AsyncIterator[ChatCompletionChunk],
    user_id: UserID,
    emit_fn: Callable[[str, str, str, bool], None] = None
) -> Message:
    """Consume streamed chunks, emit tokens, and return the assembled Message."""
    role = "assistant"
    content_parts: list[str] = []
    tool_calls_accum: dict[int, ToolCallRef] = {}

    async for chunk in chunks:
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0]["delta"]

        if "role" in delta:
            role = delta["role"]

        text = delta.get("content")
        if text:
            content_parts.append(text)
            if emit_fn:
                emit_fn(user_id, role, text, True)
            #hack?
            await asyncio.sleep(0)

        for dtc in delta.get("tool_calls", []):
            idx = dtc["index"]
            if idx not in tool_calls_accum:
                tool_calls_accum[idx] = {
                    "id": dtc.get("id", ""),
                    "type": dtc.get("type", "function"),
                    "function": {
                        "name": dtc.get("function", {}).get("name", ""),
                        "arguments": dtc.get("function", {}).get("arguments", ""),
                    },
                }
            else:
                entry = tool_calls_accum[idx]
                if "id" in dtc and dtc["id"]:
                    entry["id"] = dtc["id"]
                fn = dtc.get("function", {})
                if fn.get("name"):
                    entry["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    entry["function"]["arguments"] += fn["arguments"]

    msg: Message = {"role": role}
    content = "".join(content_parts)
    if content:
        msg["content"] = content
    else:
        msg["content"] = None
    if tool_calls_accum:
        msg["tool_calls"] = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
    return msg


# ---------------------------------------------------------------------------
# Tool-call processing
# ---------------------------------------------------------------------------


def emit(user_id: UserID, role: str, content: str, chunk: bool = False) -> None:
    responses.put_nowait((user_id, role, content, chunk))


def execute_tool_calls(
    assistant_msg: Message,
    registry: dict[str, ToolHandler],
    user_id: UserID,
    *,
    streamed: bool = False,
) -> list[Message]:
    """Return the assistant message + one tool-result message per call."""
    tool_calls: list[ToolCallRef] = assistant_msg.get("tool_calls", [])
    if not tool_calls:
        return []

    results: list[Message] = [assistant_msg]
    content = assistant_msg.get("content")
    for tc in tool_calls:
        name = tc["function"]["name"]
        args: dict[str, str] = json.loads(tc["function"]["arguments"])
        handler = registry.get(name)
        if handler is None:
            output = json.dumps({"error": f"unknown tool: {name}"})
            emit(user_id, "error", f"Unknown tool: {name}")
        else:
            args_short = ", ".join(f"{k}={v!r}" for k, v in args.items())
            if len(args_short) > 200: args_short = args_short[0:200] + "..."
            emit(user_id, "tool", f"{name}({args_short})")
            output = handler(**args)
        results.append(make_tool_result_message(tc["id"], output))
    return results


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------


def build_tools() -> tuple[list[Tool], dict[str, ToolHandler]]:
    """Build the full tools list and registry by reloading skills from disk."""
    tools: list[Tool] = [tool_from_function(run_bash), tool_from_function(vector_search), tool_from_function(download_model),
                           tool_from_function(load_model), tool_from_function(unload_model), 
                           tool_from_function(list_models)]
    registry: dict[str, ToolHandler] = {
        "run_bash": run_bash,
        "vector_search": vector_search,
        "download_model": download_model,
        "load_model": load_model_and_set,
        "unload_model": unload_model,
        "list_models": list_models,
    }
    skill_tools, skill_registry = load_skills(SKILLS_DIR)
    tools.extend(skill_tools)
    registry.update(skill_registry)
    return tools, registry


async def run_tool_loop(
    base_url: str,
    user_id: UserID,
    messages: List[Message],
    session_id: SessionID
) -> str:
    """Stream messages, handle tool calls in a loop, return final text."""
    global interrupts
    max_calls = 60
    tool_counter = 0

    while True:
        if user_id in interrupts and interrupts[user_id]:
            del interrupts[user_id]
            emit(user_id, "info", "Interrupted. What can I help you with?")
            break

        tools, registry = build_tools()
        chunks = stream_chat_completion(base_url, messages, tools, session_id)

        assistant_msg = await consume_stream(chunks, user_id, emit)
        emit(user_id, "assistant", "", False) # send a final empty message so the client knows the end of message.
        tool_msgs = execute_tool_calls(
            assistant_msg, registry, user_id, streamed=True,
        )

        if not tool_msgs:
            text = assistant_msg.get("content") or ""
            # Add timestamp when LLM responds
            assistant_msg_with_ts = {"role": "assistant", "content": text, "timestamp": datetime.now().isoformat()}
            messages.append(assistant_msg_with_ts)
            return text

        # Add timestamps to tool result messages
        tool_msgs_with_ts = []
        for tm in tool_msgs:
            tm_copy = tm.copy()
            tm_copy["timestamp"] = datetime.now().isoformat()
            tool_msgs_with_ts.append(tm_copy)
        messages.extend(tool_msgs_with_ts)
        
        tool_counter += 1
        if tool_counter > max_calls:
            emit(user_id, "info", "too many tool calls, giving up")
            text = assistant_msg.get("content") or ""
            # Add timestamp when LLM responds
            assistant_msg_with_ts = {"role": "assistant", "content": text, "timestamp": datetime.now().isoformat()}
            messages.append(assistant_msg_with_ts)
            return text
async def handle_message(
    user_input: str,
    base_url: str,
    user_id: UserID, 
    session_id: SessionID
) -> None:
    """Handle a user message event: append it, run the tool loop, emit the reply."""
    messages = get_user_messages(user_id)
    # Add timestamp when user types the message
    user_msg = make_user_message(user_input)
    user_msg["timestamp"] = datetime.now().isoformat()
    messages.append(user_msg)
    write_conversation(session_id, messages)

    reply = await run_tool_loop( # modifies messages
        base_url, user_id, messages, session_id
    )
    
    if approximate_token_count(json.dumps(messages)) > 25000:
        logger.info("Compacting conversation")
        filename = archive_conversation(session_id, messages)
        new_messages = [
            {
                "role": "assistant", 
                "content": f"The previous conversation was archived to {filename}",
                "timestamp": datetime.now().isoformat()
            },
            *messages[-4:],
        ]
        messages.clear()
        messages.extend(new_messages)

    write_conversation(session_id, messages)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
import asyncio
from typing import Dict, List, Tuple

queries = asyncio.Queue()
responses = asyncio.Queue()
interrupts = {}

async def queue_query(user_id: UserID, query: str) -> None:
    global queries
    await queries.put((user_id, query))


async def get_reponse() -> Tuple[UserID, str, str, bool]:
    return await responses.get()


async def set_interrupt(user_id: UserID) -> None:
    global interrupts
    interrupts[user_id] = True


#  USER DATABASE
def init_registry() -> Tuple[Callable[[UserID], list[Message]], Callable[[UserID, list[Messages]], None]]:
    user_registry: Dict[UserID, list[Message]] = {}

    def get_user_messages(user_id: UserID):
        if user_id not in user_registry: user_registry[user_id] = []
        return user_registry.get(user_id)
    def set_user_messages(user_id: UserID, messages: List[Message]):
        user_registry[user_id] = messages
    return get_user_messages, set_user_messages

get_user_messages, set_user_messages = init_registry()

async def main(
    on_ready: Callable = None,
    resume: bool = False
) -> None:
    global interrupts
    """Run the conversation loop, using the provided I/O callbacks."""
    # Load from environment variables if not provided
    base_url = os.getenv("LLAMA_BASE_URL")

    # Run llama.sh before starting the main server loop
    # Check if llama-server is already running
    global llama_proc
    llama_proc = None
    
    if is_llama_running(base_url):
        print("llama-server is already running, skipping start")
    else:
        # Try starting llama-server in background with fallback to setup.sh
        print("Attempting to start llama-server via start_llama.sh...")
        llama_proc = subprocess.Popen("./start_llama.sh", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        await asyncio.sleep(2)  # Wait briefly for it to start/fail without blocking
        if llama_proc.poll() is None:
            print("start_llama.sh started successfully")
        else:
            print(f"start_llama.sh exited with code {llama_proc.returncode}, trying setup.sh once...")
            subprocess.run(["./setup.sh"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    #messages: list[Message] = []

    #emit("info", f"Chat (Ctrl-C to cancel, twice to quit) - URL: {base_url}")

    session_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    if resume:
        session_id = get_last_conversation_session_id()
        #set_user_messages(read_conversation(session_id))

    session_resumed = False

    load_model_and_set(CURRENT_MODEL)

    try:
        while True:
            try:
                await asyncio.sleep(0.1)
                on_ready()
                user_id, user_input = await queries.get()
                if user_id in interrupts: del interrupts[user_id]
            except asyncio.CancelledError:
                print("Main loop cancelled", file=sys.stderr)
                break
            except KeyboardInterrupt:
                print(file=sys.stderr)
                break
            
            # load the conversation up it memory once
            if resume and not session_resumed:
                set_user_messages(user_id, read_conversation(session_id))
                session_resumed = True

            if user_input is None:
                break
            if not user_input.strip():
                continue
            try:
                await handle_message(
                    user_input, base_url, user_id, session_id
                )
            except asyncio.CancelledError:
                print("Message handling cancelled", file=sys.stderr)
                break
            except KeyboardInterrupt:
                print(file=sys.stderr)
                emit("info", "Cancelled")
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        raise
    finally:
        if llama_proc is not None:
            try:
                llama_proc.terminate()
                try:
                    llama_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    llama_proc.kill()
                print("llama-server stopped")
            except Exception as e:
                print(f"Stopping llama-server: {e}")


async def async_main(client_type: str, resume: bool = False):
    # Import client module based on command line argument
    if client_type == "terminal":
        from clients import terminal as client
    elif client_type == "discord_client":
        from clients import discord_client as client
    elif client_type == "irc_client":
        from clients import irc_client as client
    else:
        raise ValueError(f"Unknown client type: {client_type}")
    
    # Create tasks for proper cancellation
    init_task = asyncio.create_task(client.init(queue_query, get_reponse, set_interrupt))
    main_task = asyncio.create_task(main(on_ready=client.on_ready, resume=resume))
    
    try:
        await asyncio.gather(init_task, main_task)
    except asyncio.CancelledError:
        logger.info("Received cancellation, cleaning up tasks...")
        # Cancel all pending tasks
        init_task.cancel()
        main_task.cancel()
        # Wait for tasks to finish with timeout
        try:
            await asyncio.gather(init_task, main_task, return_exceptions=True)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Error in async_main: {e}")
        init_task.cancel()
        main_task.cancel()
        raise
    finally:
        # Always call stop to clean up client
        try:
            client.stop()
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")


if __name__ == "__main__":
    def load_config():
        with open("config.json") as fh:
            return json.loads(fh.read())
    config = load_config()
    import sys
    if len(sys.argv) > 1:
        config["client"] = sys.argv[1]
    resume: bool = False
    if len(sys.argv) > 2:
        resume = sys.argv[2] == "true"
    # Run the async function with proper cancellation handling
    asyncio.run(async_main(config.get("client", "terminal"), resume))
