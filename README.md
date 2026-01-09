# Claude Router

A proxy server that makes OpenRouter API and Gemini API compatible with Claude API format. Use any OpenRouter model or Gemini model with Claude Code, including full tool execution support.

## Features

- **Claude-compatible API endpoint** (`/v1/messages`)
- **Full tool execution support** (Bash, Read, Write, Edit, Glob, Grep, etc.)
- **Gemini API support** - Direct API access with your GEMINI_API_KEY (recommended)
- **OpenRouter support** - Access to hundreds of models including free options
- **Interactive model selection** on startup
- **Streaming support** for real-time responses
- **System prompt compatibility** - Auto-merges system prompts for all models
- **Passthrough architecture** - Claude Code executes tools locally while router handles API translation

## How It Works

The router acts as a translation layer between Claude Code and other AI APIs:

1. Claude Code sends requests in Claude API format to the router
2. Router translates to Gemini/OpenRouter format and forwards the request
3. Model responds with tool calls (parsed from XML/JSON format)
4. Router converts tool calls to Claude `tool_use` blocks
5. **Claude Code executes tools locally** (file operations, bash commands, etc.)
6. Claude Code sends tool results back through the router
7. Process continues until task completion

This architecture ensures full compatibility with Claude Code's tool system while using alternative AI models.

## Setup

### 1. Get API Keys

**Option A: Gemini API (Recommended)**
- Get a free API key from https://aistudio.google.com/app/apikey
- Models like `gemini-2.5-flash` have generous free tier (15 req/min, 1M context)

**Option B: OpenRouter**
- Get an API key from https://openrouter.ai/keys
- Access to hundreds of models including free options (though rate limits may be restrictive)

### 2. Set Environment Variables

Add to your `.bashrc` or `.zshrc`:

```bash
# For Gemini (recommended)
export GEMINI_API_KEY=your-gemini-api-key-here

# For OpenRouter (optional)
export OPENROUTER_API_KEY=sk-or-v1-...
```

Reload your shell:
```bash
source ~/.zshrc  # or ~/.bashrc
```

### 3. Create Claude Code Config

**Option A: Clone your existing settings**
```bash
./setup.sh
```

**Option B: Create minimal config**
```bash
python3 claude_router.py init
```

**Option C: Manual setup**

Create/edit `~/.claude/settings-router.json`:

```json
{
  "permissions": {
    "allow": ["*"],
    "defaultMode": "allow"
  },
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8083"
  }
}
```

Key fields:
- `permissions` - Tool permissions for Claude Code
- `env.ANTHROPIC_BASE_URL` - **Required**: Points to the router (default: port 8083)
- `enabledPlugins` - Optional: Copy from your existing settings if needed

## Usage

### Start the Server

```bash
python3 claude_router.py serve
```

**For Gemini models**, enter the model name directly:
```
Enter model number, model ID, or press Enter to skip:
> gemini-2.5-flash
```

**For OpenRouter models**, select interactively:
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

The server will start on `http://localhost:8083` by default.

### Use with Claude Code

```bash
claude --settings ~/.claude/settings-router.json
```

**Recommended Models:**
- `gemini-2.5-flash` - Fast, 1M context, free tier (recommended)
- `gemini-2.0-flash-thinking-exp` - Advanced reasoning (experimental)
- `meta-llama/llama-3.3-70b-instruct:free` - Free via OpenRouter
- `google/gemini-2.0-flash-exp:free` - Free via OpenRouter

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

## Troubleshooting

**"Unable to connect to API"**
- Ensure the router is running (`python3 claude_router.py serve`)
- Check that `ANTHROPIC_BASE_URL` is set to `http://localhost:8083` in settings

**"Rate limit reached"**
- OpenRouter free models have restrictive limits (often 10 req/min)
- Switch to Gemini API for better free tier (15 req/min, higher quotas)

**"Model not found"**
- For Gemini: Use stable models like `gemini-2.5-flash` or `gemini-2.0-flash-thinking-exp`
- For OpenRouter: Check available models with `python3 claude_router.py list`

**Tool execution not working**
- The router uses passthrough mode - Claude Code executes tools locally
- Ensure your settings file has proper permissions configured
- Check router logs for API translation errors

## Architecture Notes

**Why passthrough mode?**
The router doesn't execute tools server-side. Instead:
- Router translates API formats (Claude ↔ Gemini/OpenRouter)
- Claude Code executes all tools locally (file operations, bash commands)
- This ensures security and full compatibility with Claude Code's tool system

**Supported tools:**
All Claude Code tools work: Bash, Read, Write, Edit, Glob, Grep, WebFetch, TodoWrite, WebSearch, Task, and more.

## Docs

- Gemini API: https://ai.google.dev/api
- Gemini Models: https://ai.google.dev/gemini-api/docs/models/gemini
- OpenRouter Models: https://openrouter.ai/models
- OpenRouter API: https://openrouter.ai/docs

## License

MIT
