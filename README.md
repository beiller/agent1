# Agent1 Server Architecture

## Overview

Agent1 is a plugin-based AI assistant server that communicates with multiple client types (Terminal, Discord, IRC) through a unified message queue system. The server processes user queries through an LLM and executes tool calls via skills.

**Key Features:**
- Multi-platform support (Terminal, Discord, IRC)
- Plugin-based skill system for extensibility
- Conversation memory with vector search capabilities
- Async message queue architecture

## Running

### Prerequisites

You must run a llama.cpp server or use an OpenAPI compatible server. See `start_llama.sh` for an example of running locally.

### Option 1: Local Python Environment

```bash
# Create virtual environment
python -m venv venv
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py terminal
```

### Option 2: Podman Container

```bash
# WARNING: Uses host network (insecure but more secure than filesystem access)
podman-compose up
```

## Configuration

See `config.json` for available options:

- `"terminal"` - Local terminal interface
- `"discord_client"` - Discord integration (requires API key in `.env` and a Discord APP)
- `"irc_client"` - IRC integration (requires code edits for server credentials)

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
- Requires Discord API credentials

#### IRC (`irc_client.py`)
- Multi-user chat interface
- Handles private messages from multiple nicks

##### Client API

Each client has the following API provided to it:

```python
async def queue_query(query: str) -> None:
    global queries
    await queries.put(query)

async def get_response() -> Tuple[str, str, bool]:
    return await responses.get()
```

### Skill System

Skills are located in the `skills/` directory as `.md` files. Each file defines a tool with name, description, and parameters. Skills are auto-loaded at runtime via `load_skills()`.

**Built-in Skills:**
- `run_bash` - Executes shell commands
- `curl_skill` - Fetches webpage URLs
- `mux_skill` - Starts long-running commands in background tmux sessions
- `vector_search` - Searches conversation history (see below)

### Vector Search (`vector_search.py`)

A semantic search system for retrieving relevant conversation history.

**How it Works:**
1. **Chunking:** Conversations are split into paragraph-aware chunks (~300 chars each)
2. **Indexing:** TF-IDF vectorization creates searchable embeddings
3. **Search:** Cosine similarity matches queries against stored chunks
4. **Context:** Returns matching snippets with configurable context windows

**Usage:**
```python
from vector_search import vector_search

# Search for specific topics
results = vector_search(
    keyword_to_search="how to install",
    context_size=1000,  # chars of context around matches
    top_k=3,            # number of documents to return
    top_j_per_doc=2     # matches per document
)
```

**Features:**
- Paragraph-aware chunking for better semantic matching
- No caching (fresh index on every search)
- Configurable context window size
- Backslash penalty to reduce code snippet noise
- Multi-document support with ranked results

**Files Indexed:**
All `.txt` files in the `conversations/` directory are automatically included in searches.

## File Structure

```
.
├── main.py              # Entry point
├── server.py            # Core message handling
├── terminal.py          # Terminal client
├── discord_client.py    # Discord integration
├── irc_client.py        # IRC integration
├── vector_search.py     # Conversation search
├── config.json          # Configuration
├── requirements.txt     # Python dependencies
├── skills/              # Skill definitions (.md files)
│   ├── run_bash.md
│   ├── curl.md
│   └── mux.md
└── conversations/       # Stored conversation history (.txt files)
```

## Extending

### Adding New Skills

1. Create a `.md` file in `skills/` directory
2. Define tool name, description, and parameters using function-calling format
3. Implement the skill function in Python
4. Register it in the server (auto-loaded on startup)

### Adding New Clients

Implement the client interface with:
- `queue_query(query)` - Send messages to server
- `get_response()` - Receive responses from server
- Platform-specific connection handling

## License

MIT License
