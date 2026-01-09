#!/usr/bin/env python3
"""
Claude Router - A proxy server that makes OpenRouter API compatible with Claude API format.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.claude/settings-router.json")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_API_BASE}/models"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from settings-router.json"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def get_openrouter_key(config: dict = None) -> Optional[str]:
    """Get OpenRouter API key from environment"""
    return os.environ.get("OPENROUTER_API_KEY")


def get_gemini_key() -> Optional[str]:
    """Get Gemini API key from environment"""
    return os.environ.get("GEMINI_API_KEY")


def should_use_gemini(model: str) -> bool:
    """Check if we should use Gemini API directly"""
    gemini_key = get_gemini_key()
    return bool(gemini_key) and ("gemini" in model.lower() or model.startswith("google/"))


def fetch_models(api_key: Optional[str] = None) -> list:
    """Fetch available models from OpenRouter API"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request(OPENROUTER_MODELS_URL, headers=headers)
    try:
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get("data", [])
    except (HTTPError, URLError) as e:
        print(f"Error fetching models: {e}", file=sys.stderr)
        return []


def format_price(price_per_million: float) -> str:
    """Format price per million tokens"""
    if price_per_million == 0:
        return "FREE"
    elif price_per_million < 0.01:
        return f"${price_per_million:.6f}/M"
    elif price_per_million < 1:
        return f"${price_per_million:.4f}/M"
    else:
        return f"${price_per_million:.2f}/M"


def list_models(api_key: Optional[str] = None, filter_text: str = None, show_free_only: bool = False):
    """List available models with pricing information"""
    models = fetch_models(api_key)

    if not models:
        print("No models found or failed to fetch models.")
        return

    # Sort by pricing (free first, then by input price)
    def sort_key(m):
        pricing = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "0") or "0")
        return (prompt_price, m.get("id", ""))

    # Filter out meta-models with negative/dynamic pricing
    def has_valid_pricing(m):
        pricing = m.get("pricing", {})
        prompt = float(pricing.get("prompt", "0") or "0")
        return prompt >= 0

    models = [m for m in models if has_valid_pricing(m)]

    models.sort(key=sort_key)

    # Filter models
    if filter_text:
        filter_lower = filter_text.lower()
        models = [m for m in models if filter_lower in m.get("id", "").lower() or
                  filter_lower in m.get("name", "").lower()]

    if show_free_only:
        models = [m for m in models if float(m.get("pricing", {}).get("prompt", "1") or "1") == 0]

    # Print header
    print(f"\n{'Model ID':<50} {'Input Price':<14} {'Output Price':<14} {'Context':<10}")
    print("=" * 90)

    for model in models:
        model_id = model.get("id", "unknown")
        pricing = model.get("pricing", {})

        # Prices are per token, convert to per million
        prompt_price = float(pricing.get("prompt", "0") or "0") * 1_000_000
        completion_price = float(pricing.get("completion", "0") or "0") * 1_000_000
        context_length = model.get("context_length", "N/A")

        if isinstance(context_length, int):
            context_str = f"{context_length:,}"
        else:
            context_str = str(context_length)

        print(f"{model_id:<50} {format_price(prompt_price):<14} {format_price(completion_price):<14} {context_str:<10}")

    print(f"\nTotal: {len(models)} models")
    print("Docs: https://openrouter.ai/models/<model-id>")


def list_models_detailed(api_key: Optional[str] = None, model_filter: str = None):
    """List models with full details"""
    models = fetch_models(api_key)

    if model_filter:
        filter_lower = model_filter.lower()
        models = [m for m in models if filter_lower in m.get("id", "").lower()]

    for model in models:
        print(f"\n{'='*60}")
        print(f"ID: {model.get('id')}")
        print(f"Name: {model.get('name')}")
        print(f"Description: {model.get('description', 'N/A')[:100]}...")

        pricing = model.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "0") or "0") * 1_000_000
        completion_price = float(pricing.get("completion", "0") or "0") * 1_000_000

        print(f"Input: {format_price(prompt_price)} | Output: {format_price(completion_price)}")
        print(f"Context: {model.get('context_length', 'N/A'):,} tokens")

        if model.get("top_provider"):
            tp = model["top_provider"]
            print(f"Max completion: {tp.get('max_completion_tokens', 'N/A')}")

        print(f"Docs: https://openrouter.ai/models/{model.get('id')}")


# ============== Claude <-> OpenRouter Format Conversion ==============

_models_cache = {}


def get_model_info(model_id: str, api_key: str = None) -> dict:
    """Get model info from cache or fetch it"""
    global _models_cache

    if not _models_cache:
        models = fetch_models(api_key)
        _models_cache = {m.get("id"): m for m in models}

    return _models_cache.get(model_id, {})


def model_supports_system_prompt(model_id: str, api_key: str = None) -> bool:
    """Check if model supports system prompts by querying model info"""
    model_info = get_model_info(model_id, api_key)

    # Check supported_parameters if available
    supported = model_info.get("supported_parameters", [])
    if supported:
        return "system_prompt" in supported

    # Check architecture - instruction tuned models usually support it
    arch = model_info.get("architecture", {})
    instruct_type = arch.get("instruct_type")
    if instruct_type:
        return True

    # Default: assume supported (most models do)
    return True


def convert_claude_to_openrouter(claude_request: dict, model: str, api_key: str = None) -> dict:
    """Convert Claude API request format to OpenRouter format"""

    # OpenRouter uses OpenAI-compatible format
    messages = []

    system_prompt = claude_request.get("system", "")

    # Inject tools into system prompt if present
    tools = claude_request.get("tools", [])
    if tools:
        tools_instructions = format_tools_for_prompt(tools)
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{tools_instructions}"
        else:
            system_prompt = tools_instructions

    # Convert Claude messages to OpenAI format
    first_user_done = False
    for msg in claude_request.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        # Claude uses "content" as a list of content blocks or a string
        if isinstance(content, list):
            # Convert content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        pass
                    elif block.get("type") == "tool_result":
                        # Format tool result for the model
                        tool_name = block.get("tool_use", {}).get("name", "unknown")
                        result = block.get("content", "")
                        if isinstance(result, list):
                            result = "\n".join(str(r) for r in result)
                        text_parts.append(f"Tool {tool_name} result: {result}")
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        # Always prepend system prompt to first user message
        if system_prompt and role == "user" and not first_user_done:
            content = f"{system_prompt}\n\n---\n\n{content}"
            first_user_done = True

        messages.append({"role": role, "content": content})

    openrouter_request = {
        "model": model,
        "messages": messages,
        "stream": claude_request.get("stream", False),
    }

    # Map Claude parameters to OpenAI parameters
    if "max_tokens" in claude_request:
        openrouter_request["max_tokens"] = claude_request["max_tokens"]
    if "temperature" in claude_request:
        openrouter_request["temperature"] = claude_request["temperature"]
    if "top_p" in claude_request:
        openrouter_request["top_p"] = claude_request["top_p"]
    if "stop_sequences" in claude_request:
        openrouter_request["stop"] = claude_request["stop_sequences"]

    return openrouter_request


def format_tools_for_prompt(tools: list) -> str:
    """Format tools as instructions for the model to call them"""
    tools_json = json.dumps(tools, indent=2)

    # Generate examples for each tool
    tool_examples = []
    for t in tools[:5]:
        name = t.get("name")
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        args = {k: "value" for k in props}
        tool_examples.append(f'<tool_call><tool><name>{name}</name><arguments>{json.dumps(args)}</arguments></tool></tool_call>')

    examples = "\n".join(tool_examples)

    return f"""# TOOL CALLING INSTRUCTIONS

You MUST use the available tools to complete tasks. DO NOT write plans or explanations without using tools first.

## Available Tools:
{tools_json}

## CRITICAL RULES:
1. When you need information, IMMEDIATELY call the appropriate tool
2. DO NOT write implementation plans - USE TOOLS to explore and gather information
3. DO NOT describe what you would do - ACTUALLY DO IT by calling tools
4. Call tools BEFORE writing any analysis or explanation

## Tool Call Format - YOU MUST USE XML FORMAT:
<tool_call><tool><name>TOOL_NAME</name><arguments>{{"param": "value"}}</arguments></tool></tool_call>

## CORRECT Examples (XML format):

User: List files in current directory
Assistant: <tool_call><tool><name>Bash</name><arguments>{{"command": "ls -la"}}</arguments></tool></tool_call>

User: Read the README file
Assistant: <tool_call><tool><name>Read</name><arguments>{{"file_path": "README.md"}}</arguments></tool></tool_call>

User: Explore the codebase
Assistant: <tool_call><tool><name>Task</name><arguments>{{"description": "Explore codebase", "prompt": "Explore the codebase.", "subagent_type": "Explore"}}</arguments></tool></tool_call>

User: Add a todo
Assistant: <tool_call><tool><name>TodoWrite</name><arguments>{{"todos": [{{"content": "Task description", "status": "in_progress", "activeForm": "Doing task"}}]}}</arguments></tool></tool_call>

## WRONG Examples (DO NOT USE THESE):
❌ print(TodoWrite(todos=[...]))
❌ print(Task(description="...", prompt="...", subagent_type="..."))
❌ TodoWrite(todos=[...])
❌ Task(description="...")

IMPORTANT:
- Use ONLY the XML format shown above
- DO NOT use Python function calls or print statements
- Tool results will be provided automatically after you make the call"""


def parse_tool_calls_from_response(text: str) -> list:
    """Parse tool calls from model response"""
    import re
    tool_calls = []

    # Match <tool_call><tool><name>...</name><arguments>{...}</arguments></tool></tool_call>
    pattern = r'<tool_call>\s*<tool>\s*<name>\s*([^<]+)\s*</name>\s*<arguments>\s*(\{.*?\})\s*</arguments>\s*</tool>\s*</tool_call>'

    for match in re.finditer(pattern, text, re.DOTALL):
        name = match.group(1).strip()
        args_str = match.group(2)
        try:
            args = json.loads(args_str)
            tool_calls.append({
                "id": f"tool_{uuid.uuid4().hex[:8]}",
                "name": name,
                "input": args
            })
        except json.JSONDecodeError:
            pass

    # Also try simpler pattern (without <tool> wrapper)
    if not tool_calls:
        pattern2 = r'<tool_call>\s*<name>\s*([^<]+)\s*</name>\s*<arguments>\s*(\{.*?\})\s*</arguments>'
        for match in re.finditer(pattern2, text, re.DOTALL):
            name = match.group(1).strip()
            args_str = match.group(2)
            try:
                args = json.loads(args_str)
                tool_calls.append({
                    "id": f"tool_{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "input": args
                })
            except json.JSONDecodeError:
                pass

    # Try Python-style function calls: print(ToolName(...)) or ToolName(...)
    if not tool_calls:
        # Pattern to match: print(ToolName(args)) or ToolName(args)
        # Common tool names from Claude Code
        tool_names = ['TodoWrite', 'Task', 'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
                      'WebFetch', 'WebSearch', 'AskUserQuestion', 'Skill', 'EnterPlanMode']

        for tool_name in tool_names:
            # Match: print(ToolName(...)) or just ToolName(...)
            pattern = rf'(?:print\s*\()?\s*{tool_name}\s*\((.*?)\)\s*(?:\))?'
            matches = re.finditer(pattern, text, re.DOTALL)

            for match in matches:
                args_str = match.group(1).strip()
                if not args_str:
                    continue

                try:
                    # Try to parse as Python keyword arguments
                    # Convert Python syntax to JSON: key=value -> "key": value
                    # This is a simplified parser for common cases
                    args_dict = {}

                    # For TodoWrite, extract todos list
                    if tool_name == 'TodoWrite' and 'todos=' in args_str:
                        todos_match = re.search(r'todos\s*=\s*(\[.*?\])', args_str, re.DOTALL)
                        if todos_match:
                            todos_str = todos_match.group(1)
                            # Replace Python True/False with JSON true/false
                            todos_str = todos_str.replace("'", '"')
                            try:
                                todos = json.loads(todos_str)
                                args_dict['todos'] = todos
                            except:
                                pass

                    # For Task, extract parameters
                    elif tool_name == 'Task':
                        for param in ['description', 'prompt', 'subagent_type', 'model']:
                            param_match = re.search(rf'{param}\s*=\s*["\']([^"\']+)["\']', args_str)
                            if param_match:
                                args_dict[param] = param_match.group(1)

                    # For other tools, try generic parsing
                    else:
                        # Match key="value" or key='value' pairs
                        for param_match in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
                            key, value = param_match.groups()
                            args_dict[key] = value

                    if args_dict:
                        tool_calls.append({
                            "id": f"tool_{uuid.uuid4().hex[:8]}",
                            "name": tool_name,
                            "input": args_dict
                        })
                except Exception as e:
                    # If parsing fails, skip this match
                    pass

    # Also try JSON format: {"name": "...", "arguments": {...}}
    # Use a more robust approach: find JSON-like patterns and try to parse them
    if not tool_calls:
        # Find all potential JSON objects starting with {"name":
        import re
        idx = 0
        while idx < len(text):
            # Look for start of a potential tool call JSON
            match = re.search(r'\{\s*"name"\s*:', text[idx:])
            if not match:
                break

            start = idx + match.start()
            # Try to find the matching closing brace by parsing JSON
            for end in range(start + 10, min(start + 1000, len(text) + 1)):
                try:
                    potential_json = text[start:end]
                    obj = json.loads(potential_json)
                    if "name" in obj and "arguments" in obj:
                        tool_calls.append({
                            "id": f"tool_{uuid.uuid4().hex[:8]}",
                            "name": obj["name"],
                            "input": obj["arguments"]
                        })
                        idx = end
                        break
                except json.JSONDecodeError:
                    continue
            else:
                idx = start + 1

    return tool_calls


def convert_openrouter_to_claude(openrouter_response: dict, model: str) -> dict:
    """Convert OpenRouter (OpenAI) response format to Claude format"""

    choices = openrouter_response.get("choices", [])

    content = []
    stop_reason = "end_turn"

    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        text = message.get("content", "") or ""

        # Parse tool calls from the text
        tool_calls = parse_tool_calls_from_response(text)

        if tool_calls:
            # For each tool call, create a tool_use block and remove the call from text
            for call in tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["input"]
                })
            # Also include the text before tool calls (with tool calls removed)
            clean_text = strip_tool_calls(text)
            if clean_text.strip():
                content.insert(0, {"type": "text", "text": clean_text})
        elif text:
            content.append({"type": "text", "text": text})

        # Map finish_reason to Claude stop_reason
        finish_reason = choice.get("finish_reason", "stop")
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "content_filter":
            stop_reason = "end_turn"

    usage = openrouter_response.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    }


def strip_tool_calls(text: str) -> str:
    """Remove tool call XML tags from text"""
    import re
    # Remove <tool_call>...</tool_call> blocks
    text = re.sub(r'<tool_call>\s*<tool>.*?</tool>\s*</tool_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_call>\s*<name>.*?</name>\s*<arguments>.*?</arguments>\s*</tool_call>', '', text, flags=re.DOTALL)
    return text.strip()


# ============== Gemini API Support ==============

def convert_claude_to_gemini(claude_request: dict) -> dict:
    """Convert Claude API request to Gemini API format"""
    contents = []
    system_instruction = claude_request.get("system", "")

    # Add tools to system instruction
    tools = claude_request.get("tools", [])
    if tools:
        tools_instructions = format_tools_for_prompt(tools)
        if system_instruction:
            system_instruction = f"{system_instruction}\n\n{tools_instructions}"
        else:
            system_instruction = tools_instructions

    # Convert messages
    for msg in claude_request.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        # Extract text from content blocks
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "\n".join(text_parts)

        # Gemini uses "user" and "model" roles
        gemini_role = "model" if role == "assistant" else "user"

        # Add system instruction to first user message
        if gemini_role == "user" and system_instruction and not contents:
            content = f"{system_instruction}\n\n---\n\n{content}"
            system_instruction = ""  # Only add once

        contents.append({
            "role": gemini_role,
            "parts": [{"text": content}]
        })

    return {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": claude_request.get("max_tokens", 2048),
            "temperature": claude_request.get("temperature", 1.0),
            "topP": claude_request.get("top_p", 0.95),
        }
    }


def convert_gemini_to_claude(gemini_response: dict, model: str) -> dict:
    """Convert Gemini API response to Claude format"""
    candidates = gemini_response.get("candidates", [])

    content = []
    stop_reason = "end_turn"

    if candidates:
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])

        # Combine all text parts
        text = "".join(part.get("text", "") for part in parts if "text" in part)

        # Parse tool calls from text
        tool_calls = parse_tool_calls_from_response(text)

        if tool_calls:
            # Add tool_use blocks
            for call in tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["input"]
                })
            # Add remaining text
            clean_text = strip_tool_calls(text)
            if clean_text.strip():
                content.insert(0, {"type": "text", "text": clean_text})
        elif text:
            content.append({"type": "text", "text": text})

        # Map finish reason
        finish_reason = candidate.get("finishReason", "STOP")
        if finish_reason == "MAX_TOKENS":
            stop_reason = "max_tokens"

    # Get usage
    usage_metadata = gemini_response.get("usageMetadata", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage_metadata.get("promptTokenCount", 0),
            "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
        }
    }


# ============== Tool Executor ==============

def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a tool call on the server"""
    import subprocess
    import os

    if tool_name == "Bash":
        cmd = arguments.get("command", "")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout.strip() if result.stdout.strip() else result.stderr.strip() if result.stderr.strip() else "(no output)"
            return output
        except subprocess.TimeoutExpired:
            return "(command timed out)"
        except Exception as e:
            return f"(error: {str(e)})"

    elif tool_name == "Read":
        path = arguments.get("file_path", "")
        try:
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return f"(file not found: {path})"
        except Exception as e:
            return f"(error reading {path}: {str(e)})"

    elif tool_name == "Glob":
        pattern = arguments.get("pattern", "")
        try:
            import pathlib
            matches = list(pathlib.Path(".").glob(pattern))
            return "\n".join(str(m) for m in matches) if matches else "(no matches)"
        except Exception as e:
            return f"(error: {str(e)})"

    elif tool_name == "Edit":
        file_path = arguments.get("file_path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            content = content.replace(old_string, new_string)
            with open(file_path, 'w') as f:
                f.write(content)
            return "(file updated)"
        except Exception as e:
            return f"(error: {str(e)})"

    elif tool_name == "Write":
        file_path = arguments.get("file_path", "")
        text = arguments.get("text", "")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
            with open(file_path, 'w') as f:
                f.write(text)
            return "(file written)"
        except Exception as e:
            return f"(error: {str(e)})"

    else:
        return f"(unknown tool: {tool_name})"


def detect_and_execute_python(text: str) -> tuple:
    """Detect Python code blocks in text and execute them.
    Returns (executed, output, remaining_text)"""
    import re
    import subprocess
    import sys
    import os

    # Match Python code blocks (```python ... ``` or just ``` ... ```)
    pattern = r'```(?:python)?\s*(.*?)\s*```'

    matches = list(re.finditer(pattern, text, re.DOTALL))

    if not matches:
        return False, "", text

    # Execute the first code block
    match = matches[0]
    code = match.group(1)

    # Check if code tries to read a file
    read_match = re.search(r'pd\.read_csv\s*\(\s*["\']([^"\']+)["\']', code)
    if read_match:
        file_path = read_match.group(1)
        if os.path.exists(file_path):
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                # Return file contents as JSON for the model to work with
                output = f"File loaded: {file_path}\nShape: {df.shape}\nColumns: {list(df.columns)}\nFirst 5 rows:\n{df.head().to_string()}"
                remaining = text[:match.start()] + text[match.end():]
                return True, output, remaining.strip()
            except Exception as e:
                pass

    # Check if code uses pathlib.Path for directory listing
    if 'pathlib.Path' in code or '.iterdir()' in code:
        try:
            import pathlib
            path_match = re.search(r'Path\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)', code)
            if path_match:
                dir_path = pathlib.Path(path_match.group(1))
            else:
                dir_path = pathlib.Path(".")
            items = list(dir_path.iterdir())
            output = f"Directory: {dir_path}\nItems: {len(items)}\n" + "\n".join(str(i) for i in items[:20])
            remaining = text[:match.start()] + text[match.end():]
            return True, output, remaining.strip()
        except Exception as e:
            pass

    # Check for os.listdir
    if 'os.listdir' in code:
        try:
            dir_match = re.search(r'os\.listdir\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)', code)
            if dir_match:
                dir_path = dir_match.group(1)
            else:
                dir_path = "."
            items = os.listdir(dir_path)
            output = f"Directory: {dir_path}\nItems: {len(items)}\n" + "\n".join(items[:20])
            remaining = text[:match.start()] + text[match.end():]
            return True, output, remaining.strip()
        except Exception as e:
            pass

    # Remove leading import lines that Python can't handle in -c mode
    import_lines = []
    other_lines = []
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped in ('import os', 'import pathlib', 'import pandas as pd', 'import numpy as np',
                        'from pathlib import Path', 'from pathlib import Path as pathlib',
                        'import pandas', 'import numpy'):
            import_lines.append(stripped)
        elif stripped and not stripped.startswith('#'):
            other_lines.append(line)

    # Build executable code
    exec_code = '\n'.join(import_lines + other_lines)

    if not exec_code.strip():
        return False, "", text

    try:
        # Execute with timeout
        result = subprocess.run(
            [sys.executable, '-c', exec_code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="."
        )
        output = result.stdout.strip() if result.stdout.strip() else result.stderr.strip() if result.stderr.strip() else "(no output)"

        # Remove the code block from text
        remaining = text[:match.start()] + text[match.end():]
        remaining = remaining.strip()

        return True, output, remaining

    except subprocess.TimeoutExpired:
        return True, "(code execution timed out)", text[:match.start()] + text[match.end():]
    except Exception as e:
        return True, f"(execution error: {str(e)})", text[:match.start()] + text[match.end():]


def convert_python_to_tool_call(code: str, api_key: str) -> dict:
    """Use LLM to convert Python code to a tool call"""
    import urllib.request

    prompt = f"""Convert this Python code to a tool call. Choose the best tool (Bash, Read, Glob, Edit, Write):

Python code:
{code}

Output format:
<tool_call><tool><name>TOOL_NAME</name><arguments>{{"param": "value"}}</arguments></tool></tool_call>

If the code cannot be converted to a tool call, output: NONE"""

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({
                "model": "anthropic/claude-sonnet-4-20250506",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/claude-router",
                "X-Title": "Claude Router"
            }
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            result = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if result == "NONE":
                return None

            # Parse the tool call from the result
            tool_calls = parse_tool_calls_from_response(result)
            return tool_calls[0] if tool_calls else None

    except Exception as e:
        print(f"Error converting Python to tool call: {e}")
        return None


def convert_openrouter_stream_to_claude(chunk: dict, model: str, message_id: str, is_first: bool = False) -> list:
    """Convert OpenRouter streaming chunk to Claude streaming events"""
    events = []

    if is_first:
        # Send message_start event
        events.append({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        })
        # Send content_block_start
        events.append({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })

    choices = chunk.get("choices", [])
    if choices:
        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "")

        if content:
            events.append({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content}
            })

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            events.append({
                "type": "content_block_stop",
                "index": 0
            })

            stop_reason = "end_turn"
            if finish_reason == "length":
                stop_reason = "max_tokens"

            events.append({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": chunk.get("usage", {}).get("completion_tokens", 0)}
            })
            events.append({"type": "message_stop"})

    return events


# ============== HTTP Server ==============

class ClaudeRouterHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Claude Router proxy"""

    config = {}
    default_model = None

    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[{datetime.now().isoformat()}] {self.command} {self.path}")

    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version")

    def do_OPTIONS(self):
        """Handle OPTIONS request for CORS"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        print(f"GET {self.path}")
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        elif self.path == "/v1/models":
            self.handle_list_models()
        else:
            print(f"Unknown GET path: {self.path}")
            self.send_error(404)

    def handle_list_models(self):
        """Return available models"""
        api_key = get_openrouter_key(self.config)
        models = fetch_models(api_key)

        # Convert to Claude-like format
        claude_models = []
        for m in models:
            claude_models.append({
                "id": m.get("id"),
                "name": m.get("name"),
                "description": m.get("description"),
                "context_length": m.get("context_length"),
                "pricing": m.get("pricing")
            })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"data": claude_models}).encode())

    def do_POST(self):
        """Handle POST requests"""
        # Strip query string for matching
        path = self.path.split("?")[0]
        if path in ["/v1/messages", "/messages", "/api/v1/messages"]:
            self.handle_messages()
        else:
            print(f"Unknown POST path: {self.path}")
            self.send_error(404)

    def handle_messages(self):
        """Handle Claude-compatible messages endpoint"""
        try:
            print("Handling messages request...")
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            claude_request = json.loads(body.decode())
            print(f"Request model: {claude_request.get('model')}")

            # Log first message content
            messages = claude_request.get('messages', [])
            if messages:
                first_msg = messages[0].get('content', '')
                if isinstance(first_msg, str):
                    print(f"First message: {first_msg[:100]}")
                elif isinstance(first_msg, list):
                    for block in first_msg[:1]:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            print(f"First message: {block.get('text', '')[:100]}")

            api_key = get_openrouter_key(self.config)
            if not api_key:
                print("ERROR: No API key!")
                self.send_error(401, "OpenRouter API key not configured")
                return

            # Always use default model if set (overrides Claude's model)
            model = self.default_model or claude_request.get("model")
            if not model:
                self.send_error(400, "No model selected. Restart server and select a model.")
                return
            print(f"Using model: {model}")

            # Check if streaming
            is_streaming = claude_request.get("stream", False)

            # Check if tools are present - use autonomous mode for tool execution
            tools = claude_request.get("tools", [])
            has_tools = bool(tools)
            print(f"Tools in request: {len(tools)} tools, streaming: {is_streaming}")
            if tools:
                print(f"Tool names: {[t.get('name') for t in tools]}")

            # Don't use autonomous mode - let Claude Code execute tools
            # if has_tools:
            #     print("Using autonomous mode (server-side tool execution)")
            #     claude_request["stream"] = False
            #     claude_response = self.run_autonomous(claude_request, model, api_key)
            #     ...
            # else:

            if True:  # Always use passthrough mode
                # Normal passthrough mode (no tools or simple chat)
                use_gemini = should_use_gemini(model)

                if use_gemini:
                    # Use Gemini API directly
                    gemini_key = get_gemini_key()
                    if "/" in model:
                        gemini_model_name = model.split("/", 1)[1].replace(":free", "")
                    else:
                        gemini_model_name = model

                    print(f"Using Gemini API (non-autonomous): {gemini_model_name}")

                    gemini_request = convert_claude_to_gemini(claude_request)
                    url = f"{GEMINI_API_BASE}/models/{gemini_model_name}:generateContent?key={gemini_key}"

                    req = Request(
                        url,
                        data=json.dumps(gemini_request).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )

                    try:
                        with urlopen(req, timeout=120) as response:
                            data = json.loads(response.read().decode())
                            print(f"Gemini response: {json.dumps(data)[:200]}...")
                            claude_response = convert_gemini_to_claude(data, model)
                            print(f"Claude response: {json.dumps(claude_response)[:200]}...")

                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_cors_headers()
                            self.end_headers()
                            self.wfile.write(json.dumps(claude_response).encode())
                            print("Response sent successfully")
                    except HTTPError as e:
                        error_body = e.read().decode()
                        print(f"Gemini HTTPError: {error_body}", file=sys.stderr)
                        self.send_error(e.code, error_body)
                    except Exception as e:
                        print(f"Gemini error: {type(e).__name__}: {str(e)}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                        self.send_error(500, f"Internal server error: {str(e)}")
                else:
                    # Use OpenRouter
                    openrouter_request = convert_claude_to_openrouter(claude_request, model, api_key)
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://github.com/claude-router",
                        "X-Title": "Claude Router"
                    }

                    if is_streaming:
                        req = Request(
                            f"{OPENROUTER_API_BASE}/chat/completions",
                            data=json.dumps(openrouter_request).encode(),
                            headers=headers,
                            method="POST"
                        )
                        self.handle_streaming_response(req, model)
                    else:
                        req = Request(
                            f"{OPENROUTER_API_BASE}/chat/completions",
                            data=json.dumps(openrouter_request).encode(),
                            headers=headers,
                            method="POST"
                        )
                        self.handle_normal_response(req, model)

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            self.send_error(500, str(e))

    def run_autonomous(self, claude_request: dict, model: str, api_key: str) -> dict:
        """Run request with autonomous tool execution"""
        use_gemini = should_use_gemini(model)

        if use_gemini:
            gemini_key = get_gemini_key()
            # Extract model name from "google/gemini-..." format
            if "/" in model:
                gemini_model_name = model.split("/", 1)[1].replace(":free", "")
            else:
                gemini_model_name = model

            print(f"Using Gemini API directly with model: {gemini_model_name}")
        else:
            print(f"Using OpenRouter API")

        # Build conversation
        messages = []
        system = claude_request.get("system", "")
        tools = claude_request.get("tools", [])

        # Inject tools into system prompt if present
        if tools:
            tools_instructions = format_tools_for_prompt(tools)
            if system:
                system = f"{system}\n\n{tools_instructions}"
            else:
                system = tools_instructions

        request_messages = claude_request.get("messages", [])

        # Convert Claude messages to appropriate format
        first_user_done = False
        for msg in request_messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)

            # Prepend system to first user message
            if system and role == "user" and not first_user_done:
                content = f"{system}\n\n---\n\n{content}"
                first_user_done = True

            messages.append({"role": role, "content": content})

        max_iterations = 20  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                if use_gemini:
                    # Build Gemini request
                    gemini_contents = []
                    for msg in messages:
                        gemini_role = "model" if msg["role"] == "assistant" else "user"
                        gemini_contents.append({
                            "role": gemini_role,
                            "parts": [{"text": msg["content"]}]
                        })

                    gemini_request = {
                        "contents": gemini_contents,
                        "generationConfig": {
                            "maxOutputTokens": 2048,
                            "temperature": 1.0,
                        }
                    }

                    # Make request to Gemini API
                    url = f"{GEMINI_API_BASE}/models/{gemini_model_name}:generateContent?key={gemini_key}"
                    req = Request(
                        url,
                        data=json.dumps(gemini_request).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )

                    with urlopen(req, timeout=120) as response:
                        data = json.loads(response.read().decode())
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            response_text = "".join(part.get("text", "") for part in parts if "text" in part)
                        else:
                            response_text = ""
                else:
                    # Build OpenRouter request
                    openrouter_request = {
                        "model": model,
                        "messages": messages,
                    }

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://github.com/claude-router",
                        "X-Title": "Claude Router"
                    }

                    # Make request to OpenRouter
                    req = Request(
                        f"{OPENROUTER_API_BASE}/chat/completions",
                        data=json.dumps(openrouter_request).encode(),
                        headers=headers,
                        method="POST"
                    )

                    with urlopen(req, timeout=120) as response:
                        data = json.loads(response.read().decode())
                        response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # First check for and execute Python code
                py_executed, py_output, clean_text = detect_and_execute_python(response_text)
                clean_text = clean_text or ""  # Ensure clean_text is always defined

                if py_executed:
                    print(f"Iteration {iteration}: Executed Python code directly")
                    print(f"  Output: {py_output[:100]}{'...' if len(py_output) > 100 else ''}")
                    # Add assistant message first
                    messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    # Add result as user message, including remaining text
                    messages.append({
                        "role": "user",
                        "content": f"Code execution result:\n{py_output}\n\n{clean_text}".strip()
                    })
                    continue

                # Check if response contains Python code that needs LLM conversion
                python_pattern = r'```(?:python)?\s*(import\s+os|import\s+pathlib|from\s+pathlib|pathlib\.Path|\.iterdir\(\)|\.glob\(|\.read\(\)|\.write\()'
                if re.search(python_pattern, response_text):
                    # Extract the code
                    code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', response_text, re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                        print(f"Iteration {iteration}: Converting Python to tool call via LLM...")
                        # Use LLM to convert to tool call
                        tool_call = convert_python_to_tool_call(code, api_key)

                        if tool_call:
                            tool_name = tool_call.get("name")
                            arguments = tool_call.get("input", {})
                            print(f"  LLM suggested: {tool_name}({arguments})")
                            result = execute_tool_call(tool_name, arguments)
                            print(f"  Result: {result[:100]}{'...' if len(result) > 100 else ''}")

                            # Remove code from response
                            clean_text = re.sub(r'```(?:python)?\s*.*?\s*```', '', response_text, flags=re.DOTALL).strip()

                            # Add assistant message first
                            messages.append({
                                "role": "assistant",
                                "content": response_text
                            })
                            messages.append({
                                "role": "user",
                                "content": f"Tool {tool_name} result:\n{result}\n\n{clean_text}".strip()
                            })
                            continue
                        else:
                            print("  LLM could not convert to tool call")

                # Check for tool calls
                tool_calls = parse_tool_calls_from_response(response_text)

                if tool_calls:
                    print(f"Iteration {iteration}: Found {len(tool_calls)} tool call(s)")

                    # Add assistant's message first
                    messages.append({
                        "role": "assistant",
                        "content": response_text
                    })

                    # Execute each tool call and add results
                    for call in tool_calls:
                        tool_name = call.get("name")
                        arguments = call.get("input", {})
                        print(f"  Executing: {tool_name}({arguments})")
                        result = execute_tool_call(tool_name, arguments)
                        print(f"  Result: {result[:100]}{'...' if len(result) > 100 else ''}")

                        # Add result as user message
                        messages.append({
                            "role": "user",
                            "content": f"Tool {tool_name} result: {result}"
                        })
                    # Continue to next iteration
                    continue
                else:
                    # No tool calls or code, we're done
                    print(f"Iteration {iteration}: No more tool calls/code, done")
                    # Build Claude response with the clean text (code removed)
                    if clean_text:
                        claude_data = data.copy()
                        claude_data["choices"] = [{"message": {"content": clean_text}, "finish_reason": "stop"}]
                        return convert_openrouter_to_claude(claude_data, model)
                    return convert_openrouter_to_claude(data, model)

            except HTTPError as e:
                error_body = e.read().decode()
                print(f"OpenRouter error: {error_body}")
                self.send_error(e.code, error_body)
                return {"error": error_body}

        # Max iterations reached
        print(f"Max iterations ({max_iterations}) reached")
        return {
            "error": f"Max iterations ({max_iterations}) reached",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": f"(Task did not complete within {max_iterations} iterations)"}]
        }

    def handle_normal_response(self, req: Request, model: str):
        """Handle non-streaming response"""
        try:
            with urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode())

                # Log the raw response
                raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"OpenRouter response: {raw_text[:200]}")

                claude_response = convert_openrouter_to_claude(data, model)

                # Log what we're returning
                content = claude_response.get("content", [])
                if content:
                    first_block = content[0]
                    if first_block.get("type") == "text":
                        print(f"Returning text: {first_block.get('text', '')[:100]}")
                    elif first_block.get("type") == "tool_use":
                        print(f"Returning tool_use: {first_block.get('name')} with {first_block.get('input')}")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(claude_response).encode())
        except HTTPError as e:
            error_body = e.read().decode()
            print(f"OpenRouter HTTPError: {error_body}", file=sys.stderr)
            self.send_error(e.code, error_body)
        except Exception as e:
            print(f"OpenRouter error: {type(e).__name__}: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Internal server error: {str(e)}")

    def handle_streaming_response(self, req: Request, model: str):
        """Handle streaming response"""
        try:
            with urlopen(req, timeout=120) as response:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_cors_headers()
                self.end_headers()

                message_id = f"msg_{uuid.uuid4().hex[:24]}"
                is_first = True

                for line in response:
                    line = line.decode().strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        events = convert_openrouter_stream_to_claude(chunk, model, message_id, is_first)
                        is_first = False

                        for event in events:
                            event_data = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                            self.wfile.write(event_data.encode())
                            self.wfile.flush()
                    except json.JSONDecodeError:
                        continue

        except HTTPError as e:
            error_body = e.read().decode()
            print(f"OpenRouter streaming HTTPError: {error_body}", file=sys.stderr)
            self.send_error(e.code, error_body)
        except Exception as e:
            print(f"OpenRouter streaming error: {type(e).__name__}: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Internal server error: {str(e)}")


def supports_system_prompt_from_model_data(model: dict) -> bool:
    """Check if model supports system prompts from model data"""
    # Check supported_parameters if available
    supported = model.get("supported_parameters", [])
    if supported:
        return "system_prompt" in supported

    # Check architecture - instruction tuned models usually support it
    arch = model.get("architecture", {})
    instruct_type = arch.get("instruct_type")
    if instruct_type:
        return True

    # Default: assume supported (we'll prepend to user message if it fails)
    return True


def select_model_interactive(api_key: Optional[str] = None) -> Optional[str]:
    """Interactive model selection on startup"""
    print("\nFetching available models...")
    models = fetch_models(api_key)

    if not models:
        print("Failed to fetch models. Model must be specified in each request.")
        return None

    # Filter out meta-models with negative/dynamic pricing
    models = [m for m in models if float(m.get("pricing", {}).get("prompt", "0") or "0") >= 0]

    # Sort by price (free first)
    def sort_key(m):
        pricing = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "0") or "0")
        return (prompt_price, m.get("id", ""))

    models.sort(key=sort_key)

    print(f"\n{'#':<4} {'Model ID':<50} {'Input':<12} {'Output':<12}")
    print("=" * 80)

    for i, model in enumerate(models[:50], 1):  # Show first 50
        model_id = model.get("id", "unknown")
        pricing = model.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "0") or "0") * 1_000_000
        completion_price = float(pricing.get("completion", "0") or "0") * 1_000_000
        print(f"{i:<4} {model_id:<50} {format_price(prompt_price):<12} {format_price(completion_price):<12}")

    if len(models) > 50:
        print(f"\n... and {len(models) - 50} more models")

    print(f"\nDocs: https://openrouter.ai/models/<model-id>")
    print("Enter model number, model ID, or press Enter to skip:")

    try:
        choice = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not choice:
        return None

    # Check if it's a number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx].get("id")
    except ValueError:
        pass

    # Check if it's a valid model ID
    for m in models:
        if m.get("id") == choice:
            return choice

    # Partial match
    for m in models:
        if choice.lower() in m.get("id", "").lower():
            return m.get("id")

    print(f"Using model: {choice}")
    return choice


def run_server(host: str = "127.0.0.1", port: int = 8082, config: dict = None, model: str = None):
    """Run the Claude Router proxy server"""
    config = config or load_config()

    ClaudeRouterHandler.config = config
    ClaudeRouterHandler.default_model = model

    server = HTTPServer((host, port), ClaudeRouterHandler)
    print(f"\nClaude Router server running on http://{host}:{port}")
    if model:
        print(f"Default model: {model}")
    else:
        print("Model must be specified in each request")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def create_example_config():
    """Create example settings-router.json for Claude Code"""
    config = {
        "permissions": {
            "allow": ["*", "Bash"],
            "defaultMode": "bypassPermissions"
        },
        "env": {
            "ANTHROPIC_BASE_URL": "http://localhost:8082"
        }
    }

    config_dir = os.path.expanduser("~/.claude")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "settings-router.json")

    if os.path.exists(config_path):
        print(f"Config already exists at {config_path}")
        return

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created config at {config_path}")
    print("\nTo use with Claude Code:")
    print("  export OPENROUTER_API_KEY=sk-or-v1-...")
    print("  python3 claude_router.py serve")
    print("  claude --profile router")


def main():
    parser = argparse.ArgumentParser(
        description="Claude Router - OpenRouter proxy with Claude API compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                      List all models with pricing
  %(prog)s list --filter claude      Filter models containing 'claude'
  %(prog)s list --free               Show only free models
  %(prog)s list --detailed           Show detailed model info
  %(prog)s serve                     Start the proxy server
  %(prog)s serve --port 8080         Start on custom port
  %(prog)s init                      Create example config file
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--filter", "-f", help="Filter models by name")
    list_parser.add_argument("--free", action="store_true", help="Show only free models")
    list_parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed info")
    list_parser.add_argument("--api-key", "-k", help="OpenRouter API key")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the proxy server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    serve_parser.add_argument("--port", "-p", type=int, default=8082, help="Port to listen on (default: 8082)")
    serve_parser.add_argument("--config", "-c", help="Path to config file")
    serve_parser.add_argument("--model", "-m", help="Default model to use")

    # Init command
    subparsers.add_parser("init", help="Create example config file")

    args = parser.parse_args()

    if args.command == "list":
        config = load_config()
        api_key = args.api_key or get_openrouter_key(config)

        if args.detailed:
            list_models_detailed(api_key, args.filter)
        else:
            list_models(api_key, args.filter, args.free)

    elif args.command == "serve":
        config_path = args.config or DEFAULT_CONFIG_PATH
        config = load_config(config_path)

        host = args.host or config.get("server", {}).get("host", "127.0.0.1")
        port = args.port or config.get("server", {}).get("port", 8082)

        # Use --model if provided, otherwise interactive selection
        api_key = get_openrouter_key(config)
        if args.model:
            model = args.model
            print(f"Using model: {model}")
        else:
            model = select_model_interactive(api_key)

        run_server(host, port, config, model)

    elif args.command == "init":
        create_example_config()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
