# Claude Code Hooks

Hooks are shell commands that execute automatically at specific events in Claude Code's lifecycle.

## What Are Hooks?

Unlike skills/agents where Claude decides when to use them, hooks provide **deterministic control** - they run every time a specific event occurs without fail.

## Configuration

Hooks are configured in JSON settings files:
- `~/.claude/settings.json` - User settings (all projects)
- `.claude/settings.json` - Project settings (this repo)
- `.claude/settings.local.json` - Local project settings (gitignored)

**Interactive setup:**
```bash
/hooks
```

## Available Hook Events

**With matchers (for specific tools):**
- `PreToolUse` - Before tool calls (can block)
- `PermissionRequest` - When permission dialogs shown
- `PostToolUse` - After tool calls complete

**Global events:**
- `UserPromptSubmit` - When you submit a prompt
- `Stop` - When Claude finishes responding
- `SubagentStop` - When subagent completes
- `Notification` - When Claude sends notifications
- `PreCompact` - Before context compaction
- `SessionStart` - Session starts/resumes
- `SessionEnd` - Session ends

## Useful Hooks for Artifact Foundry

### Auto-format Python Snippets

Automatically format code snippets after writing:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | { read fp; [[ $fp == */snippets/*.py ]] && black \"$fp\" 2>/dev/null || true; }"
          }
        ]
      }
    ]
  }
}
```

### Protect Configuration Files

Prevent accidental edits to `.claude/` folder:
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c \"import json,sys; d=json.load(sys.stdin); p=d.get('tool_input',{}).get('file_path',''); sys.exit(2) if '.claude/' in p else sys.exit(0)\""
          }
        ]
      }
    ]
  }
}
```

### Log Learning Activity

Track what you're working on:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '\"\\(.timestamp) - \\(.tool_input.file_path)\"' >> .learning-log.txt"
          }
        ]
      }
    ]
  }
}
```

### Desktop Notifications

Get notified when Claude needs input:
```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "idle_prompt",
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' 'Awaiting your input'"
          }
        ]
      }
    ]
  }
}
```

## Hook Behavior

**Input:** Hooks receive JSON via stdin with session/event data

**Output/Exit codes:**
- Exit 0: Success
- Exit 2: Blocking error (shown to Claude)
- Other: Non-blocking error (logged only)

**Special JSON fields:**
- `permissionDecision`: "allow" | "deny" | "ask"
- `decision`: "block"
- `additionalContext`: Add context to conversation
- `continue`: false to stop Claude

## Key Features

- **Matchers** use regex (`Edit|Write`, `*` for all)
- **60-second timeout** (configurable)
- **Parallel execution** for multiple matching hooks
- **Project-aware** via `$CLAUDE_PROJECT_DIR`
- **MCP tool support** - Match with `mcp__memory__.*`

## Resources

- [Hooks Guide](https://code.claude.com/docs/en/hooks-guide.md) - Getting started
- [Hooks Reference](https://code.claude.com/docs/en/hooks.md) - Complete reference

## Notes

- Use `.claude/settings.local.json` for machine-specific hooks
- Test hooks with simple commands first
- Check stderr output if hooks aren't working
- Hooks run in your shell environment (bash/zsh)
