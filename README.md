# Itential GAIT MCP

GAIT (Git for AI Tracking) MCP Server for Itential FlowAI.

Runs directly on the IAG server with per-agent local storage. Each agent gets its own isolated subfolder.

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd itential_gait_mcp
pip install -r requirements.txt

# Run the MCP server
export GAIT_STORAGE_ROOT=/opt/gait/agents
python gait_mcp.py
```

## Architecture

```
┌─────────────────────────────────────────┐
│  FlowAI                                 │
│  └─ Calls MCP tools with agent_id       │
└──────────────────┬──────────────────────┘
                   │ MCP Protocol
                   ▼
┌─────────────────────────────────────────┐
│  IAG Server                             │
│  ├─ gait_mcp.py (this repo)             │
│  └─ /opt/gait/agents/                   │
│     ├─ agent-001/.gait/                 │
│     ├─ agent-002/.gait/                 │
│     └─ session-xyz/.gait/               │
└─────────────────────────────────────────┘
```

## FlowAI JSON Configuration

**Every tool requires `agent_id`** - this determines which subfolder stores the data.

### Initialize GAIT (auto-creates if needed)

```json
{
  "agent_id": "my-agent-session-123"
}
```

### Record a conversation turn

```json
{
  "agent_id": "my-agent-session-123",
  "user_text": "What is the status of interface Gi0/1?",
  "assistant_text": "Interface GigabitEthernet0/1 is up with IP 192.168.1.1/24",
  "artifacts": [
    {
      "path": "output/interface_status.json",
      "content": "{\"interface\": \"Gi0/1\", \"status\": \"up\"}"
    }
  ],
  "note": "network-query"
}
```

### Get conversation history

```json
{
  "agent_id": "my-agent-session-123",
  "turns": 10
}
```

## Agent ID Strategies

| Strategy | Example | Use Case |
|----------|---------|----------|
| Per-flow-execution | `flow-{{execution_id}}` | Each flow run is isolated |
| Per-session | `session-{{session_id}}` | Share context within a session |
| Per-user | `user-{{user_id}}` | User-specific persistent memory |
| Per-device | `device-{{hostname}}` | Device-specific history |

## Available Tools

### Core

| Tool | Parameters | Description |
|------|------------|-------------|
| `gait_init` | `agent_id` | Initialize GAIT for an agent |
| `gait_status` | `agent_id` | Get repo status |
| `gait_record_turn` | `agent_id`, `user_text`, `assistant_text`, `artifacts`, `note` | Record a turn |
| `gait_log` | `agent_id`, `limit` | List recent commits |
| `gait_show` | `agent_id`, `commit` | Show commit details |
| `gait_resume` | `agent_id`, `turns` | Restore conversation context |

### Memory

| Tool | Parameters | Description |
|------|------------|-------------|
| `gait_memory` | `agent_id` | List pinned items |
| `gait_pin` | `agent_id`, `commit`, `note` | Pin a commit |
| `gait_unpin` | `agent_id`, `index` | Unpin by index |
| `gait_context` | `agent_id`, `full` | Build context bundle |

### Branch & History

| Tool | Parameters | Description |
|------|------------|-------------|
| `gait_branch` | `agent_id`, `name`, `from_commit` | Create branch |
| `gait_checkout` | `agent_id`, `name` | Switch branch |
| `gait_merge` | `agent_id`, `source`, `message` | Merge branch |
| `gait_revert` | `agent_id`, `target` | Rewind history |
| `gait_squash` | `agent_id`, `last`, `message` | Squash commits |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GAIT_STORAGE_ROOT` | `/opt/gait/agents` | Base storage directory |
| `GAIT_MCP_HOST` | `0.0.0.0` | Host to bind |
| `GAIT_MCP_PORT` | `8000` | Port to listen on |

## Example FlowAI Workflow

```
1. Flow starts
   └─ gait_init(agent_id="flow-12345")

2. User asks question
   └─ gait_record_turn(agent_id="flow-12345", user_text="...", assistant_text="...")

3. Multiple turns happen...

4. Later, resume context
   └─ gait_resume(agent_id="flow-12345", turns=10)
   └─ Returns conversation history

5. Pin important context
   └─ gait_pin(agent_id="flow-12345", note="important finding")
```

## Storage Structure

```
/opt/gait/agents/
└── my-agent-session-123/
    └── .gait/
        ├── HEAD                    # Current branch ref
        ├── objects/                # Content-addressed storage
        │   ├── ab/
        │   │   └── cdef1234...     # Commit/turn objects
        │   └── ...
        └── refs/
            ├── heads/
            │   └── main            # Branch refs
            └── memory/
                └── main            # Memory manifest refs
```

## License

GPL-3.0
