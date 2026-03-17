# Agent1 Server Architecture

## Overview

Agent1 is a plugin-based AI assistant server that communicates with multiple client types (Terminal, Discord, IRC) through a unified message queue system. The server processes user queries through an LLM and executes tool calls via skills.

## Running

First you must run llama.cpp server or use an OpenAPI compatible server. See `start_llama.sh` as an example of running locally

Option 1 Make a venv `python -m venv venv` then `source ./venv/bin/activate` and `pip install -r requirements.txt` and finally `python main.py terminal`

Option 2 Run in a podman container (WARNING using host network, insecure but more secure than your filesystem access) `podman-compose up`

## Config

see `config.json`, options are 

 - "terminal"
 - "discord_client" (use discord API key in .env and create a discord APP) 
 - "irc_client" (IRC you will have to edit the code)

## High-Level Components

### Core Server (`server.py`)
The central hub that orchestrates communication between clients and the LLM.

**Responsibilities:**
- Manages HTTP transport to llama-server
- Handles conversation state (message history)
- Processes tool calls and executes skills
- Routes queries and responses through queues
- Loads and registers skill tools dynamically

**Key Data Structures:**
- `queries`: AsyncQueue - Incoming user messages from clients
- `responses`: AsyncQueue - Outgoing chat tokens/results to clients
- `messages`: List[Message] - Conversation history (persists across calls)

**Main Flow:**
```
Client → queue_query() → queries queue → server.py → handle_message() 
→ stream_chat_completion() → consume_stream() → execute_tool_calls()
→ responses queue → client
```

### Client Plugins
Three client implementations that follow the same interface pattern:

#### Terminal (`terminal.py`)
- Single-user interface (local terminal)
- Direct stdin/stdout interaction
- No user tracking needed (only one user)

#### Discord (`discord_client.py`)
- Multi-user chat interface

#### IRC (`irc_client.py`)
- Multi-user chat interface
- Handles private messages from multiple nicks


##### Client API
Each client has the following API provided to it:

python
```
async def queue_query(query: str) -> None:
    global queries
    await queries.put(query)


async def get_reponse() -> Tuple[str, str, bool]:
    return await responses.get()
```

### 4. Skill System
- Located in `skills/` directory as `.md` files
- Each file defines a tool with name, description, and parameters
- Auto-loaded at runtime via `load_skills()`
- Built in skills: `run_bash` skill executes shell commands
