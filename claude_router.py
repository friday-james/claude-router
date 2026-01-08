#!/usr/bin/env python3
"""
Claude Router - A proxy server that makes OpenRouter API compatible with Claude API format.
"""

import argparse
import json
import os
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


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from settings-router.json"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def get_openrouter_key(config: dict = None) -> Optional[str]:
    """Get OpenRouter API key from environment"""
    return os.environ.get("OPENROUTER_API_KEY")


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
    supports_system = model_supports_system_prompt(model, api_key)

    # Handle system prompt - either as system message or prepend to first user message
    prepend_system = system_prompt if (system_prompt and not supports_system) else None
    if system_prompt and supports_system:
        messages.append({"role": "system", "content": system_prompt})

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
                        # Handle image content if needed
                        pass
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        # Prepend system prompt to first user message if model doesn't support system
        if prepend_system and role == "user" and not first_user_done:
            content = f"{prepend_system}\n\n---\n\n{content}"
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


def convert_openrouter_to_claude(openrouter_response: dict, model: str) -> dict:
    """Convert OpenRouter (OpenAI) response format to Claude format"""

    choices = openrouter_response.get("choices", [])

    content = []
    stop_reason = "end_turn"

    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        text = message.get("content", "")

        if text:
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

            # Convert request format
            openrouter_request = convert_claude_to_openrouter(claude_request, model, api_key)

            # Check if streaming
            is_streaming = openrouter_request.get("stream", False)

            # Make request to OpenRouter
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/claude-router",
                "X-Title": "Claude Router"
            }

            req = Request(
                f"{OPENROUTER_API_BASE}/chat/completions",
                data=json.dumps(openrouter_request).encode(),
                headers=headers,
                method="POST"
            )

            if is_streaming:
                self.handle_streaming_response(req, model)
            else:
                self.handle_normal_response(req, model)

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            self.send_error(500, str(e))

    def handle_normal_response(self, req: Request, model: str):
        """Handle non-streaming response"""
        try:
            with urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode())
                claude_response = convert_openrouter_to_claude(data, model)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(claude_response).encode())
        except HTTPError as e:
            error_body = e.read().decode()
            print(f"OpenRouter error: {error_body}", file=sys.stderr)
            self.send_error(e.code, error_body)

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
            print(f"OpenRouter streaming error: {error_body}", file=sys.stderr)
            self.send_error(e.code, error_body)


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

        # Interactive model selection
        api_key = get_openrouter_key(config)
        model = select_model_interactive(api_key)

        run_server(host, port, config, model)

    elif args.command == "init":
        create_example_config()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
