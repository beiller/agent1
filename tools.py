#!/usr/bin/env python3
"""Tool implementations for the function-calling server."""

import json
import os
import subprocess
import urllib.request
from typing import List

from main_types import Message

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
