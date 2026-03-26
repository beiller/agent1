import functools
import itertools
import sys
import asyncio
import discord
from discord.ext import commands, tasks
import time
import os
import logging
from typing import Callable, Tuple

from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

client = None
outbox = []
inbox = []

queue_query: Callable[[str, str], None] = None
get_response: Callable[None, Tuple[str, str, str, bool]] = None
set_interrupt: Callable[[str], None] = None

import time

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

user_id_mapping = {}

@rate_limit(10, 10)
async def reply_rate_limit(user_id: str, message):
    global client, user_id_mapping
    channel_id, message_id = user_id_mapping[user_id].split(':')
    channel = await client.fetch_channel(channel_id)
    original_message = await channel.fetch_message(message_id)
    await original_message.reply(message)


def split_message(message: str, chunk_size: int = 1500) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit."""
    chunks = []
    start = 0
    while start < len(message):
        end = min(start + chunk_size, len(message))
        chunks.append(message[start:end])
        start = end
    return chunks


async def message_poll():
    global outbox
    
    while True:
        if outbox:
            for id_message in outbox:
                query_id, role, message = id_message
                # Send the full message without chunking
                # Discord allows up to 2000 characters per message
                if client and not client.is_closed() and len(message) > 0:
                    # Non assistant responses can be truncated
                    if role != "assistant" and len(message) > 200: message = message[0:200] + "..."
                    
                    # Split long messages into chunks of 1500 characters
                    chunks = split_message(message, 1500)
                    for chunk in chunks:
                        await reply_rate_limit(query_id, chunk)
                outbox = []
        await asyncio.sleep(0.1)


def get_unique_query_id(message):
    user_id = message.author.id
    message_id = message.id
    channel_id = message.channel.id
    guild_id = message.guild.id if message.guild else None
    query_id = str(channel_id) + ':' + str(message_id)
    return query_id


async def digest_user_query():
    global inbox, user_id_mapping
    while True:
        while inbox:
            message = inbox.pop()
            query_id = get_unique_query_id(message)
            user_id_mapping[message.author.id] = query_id
            await queue_query(message.author.id, message.content)
        await asyncio.sleep(0.01)


async def digest_agent_response():   
    global outbox
    message = ""
    while True:
        user_id, role, token, chunk = await get_response()
        if chunk:
            message += token
        else:
            outbox.append((user_id, role, message + token))
            message = ""


class DiscordClient(commands.Bot):
    """Custom Bot class that handles events properly"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def on_ready(self):
        logger.info(f"Discord bot ready as {self.user}")
    
    async def on_message(self, message):
        if message.author == self.user:
            return

        logger.info(f"DM from {message.author}: {message.content}")
        if message.content.lower().strip() == "stop":
            await set_interrupt(get_unique_query_id(message))
        else:
            inbox.append(message)
        # Optionally reply to confirm receipt
        #await message.reply("Message received!")
    
    async def on_raw_message_edit(self, payload):
        """Handle message edits if needed"""
        pass

async def run_discord_client():
    global client
    
    # Disable privileged intents to avoid the PrivilegedIntentsRequired error
    intents = discord.Intents.default()
    # Explicitly disable privileged intents
    intents.message_content = False  # Not needed for DMs
    intents.members = False          # Not used in this code
    
    client = DiscordClient(command_prefix="!", intents=intents)
    
    # Run in background
    await client.start(os.getenv("DISCORD_TOKEN"), reconnect=True)

def on_ready():
    pass

def stop():
    global client
    if client and not client.is_closed():
        if asyncio.get_event_loop().is_running():
            asyncio.run_coroutine_threadsafe(client.close(), asyncio.get_event_loop())
        else:
            asyncio.run(client.close())

async def init(
    _queue_query: Callable[[str, str], None], 
    _get_response: Callable[None, Tuple[str, str, str, bool]],
    _set_interrupt: Callable[[str], None]
    ):
    global queue_query, get_response, set_interrupt
    
    queue_query = _queue_query
    get_response = _get_response
    set_interrupt = _set_interrupt
    
    # Start the Discord client in a separate thread
    #import threading
    #client_thread = threading.Thread(target=run_discord_client)
    #client_thread.start()
    
    # Wait for tasks
    await asyncio.gather(
        run_discord_client(),
        digest_agent_response(), 
        digest_user_query(),
        message_poll()
    )

    client_thread.join()
