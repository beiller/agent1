import functools
import itertools
import ssl
import sys
import irc.client
import asyncio
from typing import Callable, Tuple

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

server_connected = False

def on_connect(connection, event):
    global server_connected
    server_connected = True
    return


def on_join(connection, event):
    logger.info(f"joined channel: {event.target}")


outbox = []
inbox = []
import time

queue_query: Callable[[str, str], None] = None
get_response: Callable[None, Tuple[str, str, str, bool]] = None
set_interrupt = Callable[[str], None] = None


def rate_limit(calls, period=10):
    def decorator(func):
        timestamps = []
        def wrapper(*args, **kwargs):
            now = time.time()
            timestamps[:] = [t for t in timestamps if now - t < period]
            if len(timestamps) >= calls:
                time.sleep(period - (now - timestamps[0]))
            timestamps.append(time.time())
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(10, 10)
def send_to_irc_rate_limited(connection, target, line):
    connection.privmsg(target, line)

def message_poll(connection):
    global server_connected, outbox

    if not server_connected:
        return

    if outbox:
        for message in outbox:
            # Each message should be a tuple (user_id, full_message)
            user_id, full_message = message
            logger.info(f"Sending to {user_id}: {full_message[:50]}...")
            for line in full_message.split("\n"):
                if len(line.encode('utf-8')) <= 400:
                    send_to_irc_rate_limited(connection, user_id, line)
                else:
                    # Split long messages into chunks
                    bytes_line = line.encode('utf-8')
                    pos = 0
                    while pos < len(bytes_line):
                        chunk = bytes_line[pos:pos+400].decode('utf-8')
                        send_to_irc_rate_limited(connection, user_id, chunk)
                        pos += 400
        outbox.clear()


async def digest_user_query():
    global inbox
    while True:
        while inbox:
            user_id, message = inbox.pop(0)
            logger.info(f"Processing query from {user_id}: {message[:50]}...")
            await queue_query(user_id, message)
        await asyncio.sleep(0.01)


async def digest_agent_response():   
    global outbox
    message = ""
    while True:
        user_id, role, token, chunk = await get_response()
        if chunk:
            message += token
        else:
            outbox.append((user_id, message + token))
            message = ""


def on_disconnect(connection, event):
    logger.info("disconnected")


def on_privmsg(connection, event):
    """Handle incoming private messages with unique user ID (nick)"""
    global inbox
    user_id = event.source.nick  # This is the unique identifier for the user
    message = event.arguments[0]
    
    logger.info(f"DM from {user_id}: {message}")
    inbox.append((user_id, message))


def run_irc_reactor_process():
    global target

    target = 'william_'

    context = ssl.create_default_context()
    wrapper = functools.partial(context.wrap_socket, server_hostname='irc.libera.chat')
    ssl_factory = irc.connection.Factory(wrapper=wrapper)
    reactor = irc.client.Reactor()
    try:
        c = reactor.server().connect(
            'irc.libera.chat', 6697, 'sdfdsf333', connect_factory=ssl_factory
        )
    except irc.client.ServerConnectionError:
        logger.error(sys.exc_info()[1])
        raise SystemExit(1) from None

    c.add_global_handler("welcome", on_connect)
    c.add_global_handler("join", on_join)
    c.add_global_handler("disconnect", on_disconnect)
    c.add_global_handler("privmsg", on_privmsg)
    c.add_global_handler("pubmsg", on_privmsg)

    reactor.scheduler.execute_every(0.2, lambda: message_poll(c))

    reactor.process_forever()


def on_ready():
    pass


def stop():
    pass

async def init(
    _queue_query: Callable[[str, str], None], 
    _get_response: Callable[None, Tuple[str, str, str, bool]],
    _set_interrupt: Callable[[str], None]
    ):
    global queue_query, get_response, set_interrupt
    queue_query = _queue_query
    get_response = _get_response
    set_interrupt = _set_interrupt

    await asyncio.gather(
        digest_agent_response(), 
        digest_user_query(), 
        asyncio.get_event_loop().run_in_executor(None, run_irc_reactor_process)
    )
