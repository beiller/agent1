from typing import Callable, get_type_hints, Protocol, TypedDict


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
