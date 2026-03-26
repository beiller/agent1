#!/usr/bin/env python3
"""Tool implementations for the function-calling server."""

import json
import os
import subprocess
import urllib.request
from typing import List
import pathlib
from main_types import Message




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



# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


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


def load_model(model_name: str) -> str:
    """Load a specific model into llama-server router mode via its API.
    
    Requires llama-server to be running in router mode (started without --model flag).
    The model must be discoverable via --models-dir or preset configuration.
    
    Args:
        model_name: Name or path of the model to load
        
    Returns:
        JSON string with success status and details
    """
    try:
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



def load_skills(
    skills_dir: str | pathlib.Path
) -> tuple[list[Tool], dict[str, ToolHandler]]:
    """Scan skills_dir for .md files and return (tools, registry) for each."""
    skills_dir = pathlib.Path(skills_dir)

    tools = []
    registry = {}
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
