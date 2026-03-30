

import base64
import re
from io import BytesIO
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog
from textual.events import Paste
import asyncio
import signal
import logging

from typing import Callable

logger = logging.getLogger()

queue_query = None
get_response = None
set_interrupt = None
app = None
_shutdown_event = asyncio.Event()


from textual.widgets import Static, Input, Markdown
from textual.containers import VerticalScroll



class ChatApp(App):
    STATE_STREAMING = "streaming"
    STATE_IDLE = "idle"
    async def on_mount(self):
        # This ensures the input is ready for typing immediately
        self.query_one(Input).focus()

        container = self.query_one("#chat_container")
        await container.mount(Static(f"[bold cyan]✦  C H A T[/]"))
        await container.mount(Static(f"[grey]Local AI Assistant[/]"))

        self.state = ChatApp.STATE_IDLE

        self.run_worker(write_output())

    def compose(self) -> ComposeResult:
        # Use a scrollable container for the chat history
        yield VerticalScroll(id="chat_container")
        yield Input(placeholder="Type here...")


    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip(): return
        container = self.query_one("#chat_container")
        await container.mount(Static(f"[bold cyan]You:[/][white] {event.value}[/]"))
        container.scroll_end(animate=False)
        await queue_query("0", event.value)
        event.input.value = ""


    async def debug(self, text):
        container = self.query_one("#chat_container")
        await container.mount(Static(text))
        container.scroll_end(animate=False)


    async def create_container(self, markdown=False):
        container = self.query_one("#chat_container")
        widget = Markdown("") if markdown else Static("")
        await container.mount(widget)
        self.current_response = widget


    async def on_llm_text(self, user_id, role, token, chunk):
        """Now you can append tokens without a newline!"""
        if self.state == ChatApp.STATE_IDLE:
            if chunk:
                self.state = ChatApp.STATE_STREAMING
                self.current_text = ""
                await self.create_container(markdown=True)
            else:
                await self.create_container(markdown=False)
        
        if self.state == ChatApp.STATE_STREAMING and not chunk and token == "":
            self.state = ChatApp.STATE_IDLE
            return

        if self.state == ChatApp.STATE_STREAMING:
            self.current_text += token
            new_content = self.current_text
        else:
            new_content = f"[bold magenta]{role} 🤖:[/][white] {token}[/]"

        self.current_response.update(new_content)
        container = self.query_one("#chat_container")
        container.scroll_end(animate=False)

    # async def on_paste(self, event: Paste) -> None:
    #     file_path = event.text.strip().replace("file://", "")
    #     await self.debug(file_path)
    #     return


async def write_output():
    """Write output responses. Handles shutdown gracefully."""
    global app
    logger.info("Starting output writer...")
    try:
        while not _shutdown_event.is_set():
            try:
                # Use a timeout to check shutdown event periodically
                user_id, role, token, chunk = await asyncio.wait_for(
                    get_response(),
                    timeout=0.5
                )
                asyncio.run_coroutine_threadsafe(app.on_llm_text(user_id, role, token, chunk), asyncio.get_event_loop())
            except asyncio.TimeoutError:
                # Check if we should shutdown
                continue
            except asyncio.CancelledError:
                logger.info("Output writer cancelled")
                break
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Error in write_output: {e}")
        raise


def on_ready():
    #print_prompt()
    pass


def stop():
    """Signal shutdown and clean up terminal state."""
    logger.info("Stopping terminal...")
    _shutdown_event.set()


def on_ctrl_c():
    global queue_query, get_response, set_interrupt
    #print("\nCtrl+C ignored, still running. Use kill or another method to stop.")
    asyncio.run_coroutine_threadsafe(set_interrupt("0"), loop)


async def run_ui():
    global app
    app = ChatApp()
    await app.run_async()


async def init(_queue_query: Callable, _get_response: Callable, _set_interrupt: Callable):
    """Initialize the terminal client and start I/O tasks."""
    global queue_query, get_response, set_interrupt
    queue_query = _queue_query
    get_response = _get_response
    set_interrupt = _set_interrupt
    
    # Start both tasks with proper cancellation handling
    try:
        await run_ui()
    except asyncio.CancelledError:
        logger.info("Terminal init cancelled")
        stop()
        raise
