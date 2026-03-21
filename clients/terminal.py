"""Terminal plugin – provides display and input callbacks for the server."""

from __future__ import annotations

import readline  # noqa: F401 – enables arrow-key / history in input()
import sys
import asyncio
import logging

logger = logging.getLogger()

queue_query: Callable[str] = None
get_response: Callable[None] = None
_shutdown_event = asyncio.Event()


def _ansi(code: str) -> str:
    return f"\033[{code}m"

RESET = _ansi("0")
BOLD = _ansi("1")
DIM = _ansi("2")
BRIGHT_CYAN = _ansi("96")
BRIGHT_GREEN = _ansi("92")
YELLOW = _ansi("33")
BRIGHT_MAGENTA = _ansi("95")
RED = _ansi("31")
GRAY = _ansi("90")

UP_1_LINE = "\033[A"

_STYLE_MAP: dict[str, tuple[str, str]] = {
    "assistant": (BRIGHT_CYAN, "assistant"),
    "info": (YELLOW, "info"),
    "skill": (BRIGHT_GREEN, "skill"),
    "tool": (BRIGHT_MAGENTA, "tool"),
    "error": (RED, "error"),
}

SEP = f"{GRAY}{'─' * 40}{RESET}"


def emit(prefix: str, text: str) -> None:
    """Print a styled, prefixed message to stderr."""
    color, label = _STYLE_MAP.get(prefix, (RESET, prefix))

    if prefix == "assistant":
        print(
            f"{color}┌── 🤖 {BOLD}{label}{RESET}{color} ─────────────────{RESET}",
            file=sys.stderr,
        )
        for line in text.splitlines():
            print(f"{color}│{RESET} {line}", file=sys.stderr)
        print(f"{color}└───────────────────────────────{RESET}", file=sys.stderr)
    else:
        tag = f"{color}{BOLD}▸ {label}{RESET}"
        lines = text.splitlines()
        first, rest = lines[0], lines[1:]
        print(f" {tag} {color}{first}{RESET}", file=sys.stderr)
        indent = "    "
        for line in rest:
            print(f" {indent}{color}{line}{RESET}", file=sys.stderr)


def print_prompt():
    prompt = "> "
    sys.stderr.write(f"{BOLD}{BRIGHT_GREEN}{prompt}{RESET}")
    sys.stderr.flush()

async def ainput(string: str) -> str:
    await asyncio.get_event_loop().run_in_executor(
            None, lambda s=string: sys.stdout.write(s+' '))
    return await asyncio.get_event_loop().run_in_executor(
            None, sys.stdin.readline)

async def read_input() -> str | None:
    """Read user input and queue queries. Handles shutdown gracefully."""
    logger.info("Starting input reader...")
    try:
        while not _shutdown_event.is_set():
            try:
                user_prompt = await ainput("")
                await queue_query("0", user_prompt)
            except asyncio.TimeoutError:
                # Check if we should shutdown
                if _shutdown_event.is_set():
                    break
                continue
            except EOFError:
                logger.info("EOF received, shutting down input reader")
                print(f"\n{SEP}", file=sys.stderr)
                break
            except asyncio.CancelledError:
                logger.info("Input reader cancelled")
                break
    except Exception as e:
        logger.error(f"Error in read_input: {e}")
        raise
    finally:
        print(f"{RESET}\n", file=sys.stderr)


async def write_output():
    """Write output responses. Handles shutdown gracefully."""
    logger.info("Starting output writer...")
    try:
        while not _shutdown_event.is_set():
            try:
                # Use a timeout to check shutdown event periodically
                user_id, role, token, chunk = await asyncio.wait_for(
                    get_response(),
                    timeout=0.5
                )
                color, label = _STYLE_MAP.get(role, (RESET, role))
                if token != "":
                    sys.stderr.write(f"{color}{token}{RESET}")
                if not chunk:
                    sys.stderr.write(f"\n")

                sys.stderr.flush()
            except asyncio.TimeoutError:
                # Check if we should shutdown
                continue
            except asyncio.CancelledError:
                logger.info("Output writer cancelled")
                break
    except Exception as e:
        logger.error(f"Error in write_output: {e}")
        raise
    finally:
        sys.stderr.write(f"{RESET}\n")
        sys.stderr.flush()


def print_banner() -> None:
    print(file=sys.stderr)
    print(f"  {BOLD}{BRIGHT_CYAN}✦  C H A T{RESET}", file=sys.stderr)
    print(f"  {DIM}Local AI Assistant{RESET}", file=sys.stderr)
    print(file=sys.stderr)


def on_ready():
    print_prompt()


def stop():
    """Signal shutdown and clean up terminal state."""
    logger.info("Stopping terminal...")
    _shutdown_event.set()
    sys.stderr.write(f"{RESET}\n")
    print(f"{RESET}", file=sys.stderr)
    # Reset terminal colors
    sys.stderr.write("\033[0m")
    sys.stderr.flush()


async def init(_queue_query: Callable[str], _get_response: Callable[None]):
    """Initialize the terminal client and start I/O tasks."""
    global queue_query, get_response
    queue_query = _queue_query
    get_response = _get_response
    print_banner()
    
    # Start both tasks with proper cancellation handling
    try:
        await asyncio.gather(write_output(), read_input())
    except asyncio.CancelledError:
        logger.info("Terminal init cancelled")
        stop()
        raise
