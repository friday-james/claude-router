#!/bin/bash
# Clone Claude settings to settings-router.json with OpenRouter proxy config

CLAUDE_DIR="$HOME/.claude"
SOURCE="$CLAUDE_DIR/settings-local.json"
TARGET="$CLAUDE_DIR/settings-router.json"
PORT="${1:-8082}"

# Check if source exists
if [ ! -f "$SOURCE" ]; then
    echo "No settings-local.json found at $SOURCE"
    echo "Creating minimal config instead..."

    cat > "$TARGET" << EOF
{
  "permissions": {
    "allow": ["*", "Bash"],
    "defaultMode": "bypassPermissions"
  },
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:$PORT"
  }
}
EOF
    echo "Created $TARGET"
    exit 0
fi

# Check if target already exists
if [ -f "$TARGET" ]; then
    read -p "settings-router.json already exists. Overwrite? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Clone and modify settings
echo "Cloning $SOURCE to $TARGET..."

# Use python to properly merge JSON
python3 << EOF
import json

with open("$SOURCE", "r") as f:
    config = json.load(f)

# Ensure env section exists
if "env" not in config:
    config["env"] = {}

# Set the proxy URL
config["env"]["ANTHROPIC_BASE_URL"] = "http://localhost:$PORT"

# Remove any existing Anthropic API key (we use OpenRouter)
config["env"].pop("ANTHROPIC_API_KEY", None)

with open("$TARGET", "w") as f:
    json.dump(config, f, indent=2)

print("Done!")
EOF

echo ""
echo "Created: $TARGET"
echo ""
echo "Usage:"
echo "  1. export OPENROUTER_API_KEY=sk-or-v1-..."
echo "  2. python3 claude_router.py serve"
echo "  3. claude --profile router"
