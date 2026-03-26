import pathlib
import json
from datetime import datetime
from typing import List

from main_types import Message, SessionID

CONVERSATION_DIR = './conversations/'


def approximate_token_count(text: str) -> int:
    return int(len(text) / 4)


def archive_conversation(session_id: str, messages: List[Message]) -> str:
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + session_id + ".txt"
    archive_text = ""
    for message in messages:
        if 'timestamp' in message and message['timestamp']:
            archive_text += '[' + message['timestamp'] + '] '
        archive_text += message['role'] + ":\n"
        if 'content' in message and message['content']:
            archive_text += message['content'] + "\n"
        if 'tool_calls' in message and message['tool_calls']:
            archive_text += json.dumps(message['tool_calls']) + "\n"
        archive_text += "\n"

    with open(CONVERSATION_DIR + filename, 'w') as fh:
        fh.write(archive_text)

    return CONVERSATION_DIR + filename


def write_conversation(session_id: str, messages: List[Message]) -> None:
    with open(CONVERSATION_DIR + session_id + '.json', 'w') as fh:
        json.dump(messages, fh)


def read_conversation(session_id: str) -> List[Message]:
    with open(CONVERSATION_DIR + session_id + '.json') as fh:
        return json.load(fh)


def get_last_conversation_session_id() -> SessionID:
    files = [f for f in pathlib.Path(CONVERSATION_DIR).iterdir() 
             if f.is_file() and f.name.endswith(".json")]
    session_id: SessionID = max(files, key=lambda f: f.stat().st_mtime).name.split('/')[-1].split('.')[0]
    return session_id
