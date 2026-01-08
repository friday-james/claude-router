# Claude Router

A proxy server that makes OpenRouter API compatible with Claude API format. Use any OpenRouter model with Claude Code.

## Features

- Claude-compatible API endpoint (`/v1/messages`)
- Interactive model selection on startup
- Lists all models with pricing (input/output per million tokens)
- Streaming support
- Auto-prepends system prompt to first user message for models that don't support system prompts natively

## Setup

1. Get an OpenRouter API key from https://openrouter.ai/keys

2. Set environment variable:
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

3. Create Claude Code config:
```bash
python3 claude_router.py init
```

This creates `~/.claude/settings-router.json`:
```json
{
  "permissions": {
    "allow": ["*", "Bash"],
    "defaultMode": "bypassPermissions"
  },
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8082"
  }
}
```

## Usage

### Start the server

```bash
python3 claude_router.py serve
```

Select a model interactively:
```
#    Model ID                                           Input        Output
================================================================================
1    meta-llama/llama-3.3-70b-instruct:free             FREE         FREE
2    google/gemini-2.0-flash-exp:free                   FREE         FREE
3    deepseek/deepseek-r1-0528:free                     FREE         FREE
...

Enter model number, model ID, or press Enter to skip:
> 1
```

### Use with Claude Code

```bash
claude --profile router
```

### List models

```bash
# List all models with pricing
python3 claude_router.py list

# Filter by name
python3 claude_router.py list --filter llama

# Show only free models
python3 claude_router.py list --free

# Detailed view
python3 claude_router.py list --detailed --filter gpt
```

## CLI Reference

```
claude_router.py list [--filter NAME] [--free] [--detailed]
claude_router.py serve [--host HOST] [--port PORT]
claude_router.py init
```

## Docs

- OpenRouter Models: https://openrouter.ai/models
- OpenRouter API: https://openrouter.ai/docs

## License

MIT
