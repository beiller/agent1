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
from vector_search import vector_search
import random
import string
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


CURRENT_MODEL = "Qwen_Qwen3.5-27B-Q4_1"

# ---------------------------------------------------------------------------
# Types – Chat Completions API
# ---------------------------------------------------------------------------


class PropertyDef(TypedDict, total=False):
    type: str
    description: str


class FunctionParameters(TypedDict):
    type: str
    properties: dict[str, PropertyDef]
    required: list[str]


class FunctionDef(TypedDict):
    name: str
    description: str
    parameters: FunctionParameters


class Tool(TypedDict):
    type: str
    function: FunctionDef


class FunctionCallRef(TypedDict):
    name: str
    arguments: str


class ToolCallRef(TypedDict):
    id: str
    type: str
    function: FunctionCallRef


class Message(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[ToolCallRef]
    tool_call_id: str


class Choice(TypedDict):
    index: int
    message: Message
    finish_reason: str


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ChatCompletionRequest(TypedDict, total=False):
    model: str
    messages: list[Message]
    tools: list[Tool]
    tool_choice: str
    stream: bool


# ---------------------------------------------------------------------------
# Types – Streaming (SSE) chunks
# ---------------------------------------------------------------------------


class DeltaToolCallFunction(TypedDict, total=False):
    name: str
    arguments: str


class DeltaToolCall(TypedDict, total=False):
    index: int
    id: str
    type: str
    function: DeltaToolCallFunction


class Delta(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[DeltaToolCall]


class StreamChoice(TypedDict):
    index: int
    delta: Delta
    finish_reason: str | None


class ChatCompletionChunk(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[StreamChoice]


# ---------------------------------------------------------------------------
# Protocol – tool handler contract
# ---------------------------------------------------------------------------


class ToolHandler(Protocol):
    def __call__(self, **kwargs: str) -> str: ...


# Callback signatures for the client interface.
Emitter = Callable[[str, str], None]
EmitToken = Callable[[str], None]
ReadInput = Callable[[], str | None]


class StreamWriterProto(Protocol):
    def write_token(self, token: str) -> None: ...
    def finish(self) -> None: ...


StreamWriterFactory = Callable[[], StreamWriterProto]


# ---------------------------------------------------------------------------
# Pure constructors
# ---------------------------------------------------------------------------


def make_tool(
    name: str,
    description: str,
    parameters: FunctionParameters,
) -> Tool:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


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
    tools: list[Tool]
) -> AsyncIterator[ChatCompletionChunk]:
    """Send a streaming chat completion request, yielding SSE chunks asynchronously."""
    system_content = SYSTEM_MD.read_text() if SYSTEM_MD.exists() else ""
    msgs = [make_system_message(system_content)] + [
        m for m in messages if m.get("role") != "system"
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
            emit(user_id, "tool", f"{name}({args_short})")
            output = handler(**args)
        results.append(make_tool_result_message(tc["id"], output))
    return results


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------


def build_tools() -> tuple[list[Tool], dict[str, ToolHandler]]:
    """Build the full tools list and registry by reloading skills from disk."""
    tools: list[Tool] = [tool_from_function(run_bash), tool_from_function(vector_search), 
                           tool_from_function(load_model), tool_from_function(unload_model), 
                           tool_from_function(list_models)]
    registry: dict[str, ToolHandler] = {
        "run_bash": run_bash,
        "vector_search": vector_search,
        "load_model": load_model,
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
    messages: List[Message]
) -> str:
    """Stream messages, handle tool calls in a loop, return final text."""
    max_calls = 60
    tool_counter = 0

    #messages: List[Message] = get_user_messages(user_id)

    while True:
        tools, registry = build_tools()
        chunks = stream_chat_completion(base_url, messages, tools)

        assistant_msg = await consume_stream(chunks, user_id, emit)
        emit(user_id, "assistant", "", False)
        USE_STREAMED=True
        tool_msgs = execute_tool_calls(
            assistant_msg, registry, user_id, streamed=USE_STREAMED,
        )

        if not tool_msgs:
            text = assistant_msg.get("content") or ""
            messages.append({"role": "assistant", "content": text})
            return text

        messages.extend(tool_msgs)
        tool_counter += 1
        if tool_counter > max_calls:
            emit(user_id, "info", "too many tool calls, giving up")
            text = assistant_msg.get("content") or ""
            messages.append({"role": "assistant", "content": text})
            return text



def approximate_token_count(text: str) -> int:
    # Rough estimate: ~4 characters per token for English text
    return int(len(text) / 4)
 
async def handle_message(
    user_input: str,
    base_url: str,
    user_id: UserID, 
    session_id: SessionID
) -> None:
    """Handle a user message event: append it, run the tool loop, emit the reply."""
    messages = get_user_messages(user_id)
    messages.append(make_user_message(user_input))
    write_conversation(session_id, messages)

    reply = await run_tool_loop( # modifies messages
        base_url, user_id, messages
    )
    
    if approximate_token_count(json.dumps(messages)) > 25000:
        logger.info("Compacting conversation")
        filename = archive_conversation(session_id, messages)
        new_messages = [
            {"role": "assistant", "content": f"The previous conversation was archived to {filename}"},
            *messages[-4:],
        ]
        messages.clear()
        messages.extend(new_messages)

    write_conversation(session_id, messages)



# ---------------------------------------------------------------------------
# Skill loader – turns skills/*.md into tools
# ---------------------------------------------------------------------------


def load_skills(
    skills_dir: str | pathlib.Path,
) -> tuple[list[Tool], dict[str, ToolHandler]]:
    """Scan skills_dir for .md files and return (tools, registry) for each."""
    skills_dir = pathlib.Path(skills_dir)
    tools: list[Tool] = []
    registry: dict[str, ToolHandler] = {
        "run_bash": run_bash,
        "vector_search": vector_search,
        "load_model": load_model,}

    if not skills_dir.is_dir():
        return tools, registry

    for md_file in sorted(skills_dir.glob("*.md")):
        text = md_file.read_text()
        lines = text.splitlines()
        description = lines[0].strip() if lines else md_file.stem
        body = "\n".join(lines[1:]).strip()
        skill_name = md_file.stem + "_skill"

        tool = make_tool(
            name=skill_name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"},
                },
                "required": ["command"],
            },
        )

        def _make_handler(content: str):
            def handler(*, command: str) -> str:
                return content

            return handler

        tools.append(tool)
        registry[skill_name] = _make_handler(body)

    return tools, registry


# ---------------------------------------------------------------------------
# Tool implementations
CONVERSATION_DIR = './conversations/'

def load_model(model_name: str) -> str:
    """Load a specific model into llama-server router mode via its API.
    
    Requires llama-server to be running in router mode (started without --model flag).
    The model must be discoverable via --models-dir or preset configuration.
    
    Args:
        model_name: Name or path of the model to load
        
    Returns:
        JSON string with success status and details
    """
    global CURRENT_MODEL
    unload_model(CURRENT_MODEL)
    base_url = os.getenv("LLAMA_BASE_URL")
    CURRENT_MODEL = model_name 
    try:
        # llama-server router mode load endpoint
        url = f"{base_url}/models/load"
        data = json.dumps({"model": model_name}).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return json.dumps({"success": True, "model": model_name, "details": result})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else "Unknown error"
        return json.dumps({"success": False, "http_status": e.code, "error": error_body})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def unload_model(model_name: str) -> str:
    """Unload a specific model from llama-server router mode via its API.
    
    Requires llama-server to be running in router mode.
    
    Args:
        model_name: Name or path of the model to unload
        
    Returns:
        JSON string with success status and details
    """
    base_url = os.getenv("LLAMA_BASE_URL")
    try:
        url = f"{base_url}/models/unload"
        data = json.dumps({"model": model_name}).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return json.dumps({"success": True, "model": model_name, "details": result})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else "Unknown error"
        return json.dumps({"success": False, "http_status": e.code, "error": error_body})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def list_models() -> str:
    """List all available models in llama-server router mode.
    
    Requires llama-server to be running in router mode.
    Returns model names, statuses, and metadata.
    
    Returns:
        JSON string with list of models and their status
    """
    base_url = os.getenv("LLAMA_BASE_URL")
    try:
        url = f"{base_url}/models"
        
        req = urllib.request.Request(
            url,
            method="GET"
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})



def archive_conversation(session_id: str, messages: List[Message]):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + session_id + ".txt"
    archive_text = ""
    for message in messages:
        if message['role'] in ['tool', ] : continue
        archive_text += message['role'] + ":\n"
        if 'content' in message and message['content']: archive_text += message['content'] + "\n"
        if 'tool_calls' in message and message['tool_calls']: archive_text += json.dumps(message['tool_calls']) + "\n"
        archive_text += "\n"

    with open(CONVERSATION_DIR+filename, 'w') as fh:
        fh.write(archive_text)

    return CONVERSATION_DIR+filename


def write_conversation(session_id: str, messages: List[Message]):
    with open(CONVERSATION_DIR+session_id+'.json', 'w') as fh:
        json.dump(messages, fh)


def read_conversation(session_id: str) -> List[Message]:
    with open(CONVERSATION_DIR+session_id+'.json') as fh:
        return json.load(fh)


def get_last_conversation_session_id() -> SessionID:
    files = [f for f in pathlib.Path(CONVERSATION_DIR).iterdir() if f.is_file() and f.name.endswith(".json")]
    session_id: SessionID = max(files, key=lambda f: f.stat().st_mtime).name.split('/')[-1].split('.')[0]
    logger.info("Resuming session: " + session_id)
    return session_id


def run_bash(*, command: str) -> str:
    """Run a bash command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        return json.dumps(
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "error": f"Command execution failed: {str(e)}",
            }
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
import asyncio
from typing import Dict, List, Tuple

queries = asyncio.Queue()
responses = asyncio.Queue()


async def queue_query(user_id: UserID, query: str) -> None:
    global queries
    await queries.put((user_id, query))


async def get_reponse() -> Tuple[str, str, str, bool]:
    return await responses.get()

#  USER DATABASE
UserID = str
SessionID = str
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
    """Run the conversation loop, using the provided I/O callbacks."""
    _, skill_registry = load_skills(SKILLS_DIR)
    #for name in skill_registry: emit("skill", name)

    # Load from environment variables if not provided
    base_url = os.getenv("LLAMA_BASE_URL")
    
    #messages: list[Message] = []

    #emit("info", f"Chat (Ctrl-C to cancel, twice to quit) - URL: {base_url}")

    session_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    if resume:
        session_id = get_last_conversation_session_id()
        #set_user_messages(read_conversation(session_id))

    session_resumed = False

    load_model(CURRENT_MODEL)

    try:
        while True:
            try:
                await asyncio.sleep(0.1)
                on_ready()
                user_id, user_input = await queries.get()
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


async def async_main(client_type: str, resume: bool = False):
    # Import client module based on command line argument
    if client_type == "terminal":
        import terminal
    elif client_type == "discord_client":
        import discord_client as terminal
    elif client_type == "irc_client":
        import irc_client as terminal
    else:
        raise ValueError(f"Unknown client type: {client_type}")
    
    # Create tasks for proper cancellation
    init_task = asyncio.create_task(terminal.init(queue_query, get_reponse))
    main_task = asyncio.create_task(main(on_ready=terminal.on_ready, resume=resume))
    
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
        # Always call stop to clean up terminal
        try:
            terminal.stop()
        except Exception as e:
            logger.warning(f"Error during terminal cleanup: {e}")


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

