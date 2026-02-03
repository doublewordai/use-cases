# Async Agents: Recursive Tool-Calling Research via Batch API

A single root agent drives the entire research process — breaking a topic into sub-queries, spawning parallel sub-agents, and synthesizing a final report. Sub-agents can recursively spawn their own sub-agents, creating a tree of autonomous researchers. All agents at all depths run in parallel within each batch round, and the model decides everything: what to search, what to read, when to delegate, and when to write the report.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## How It Works

The model controls the entire research workflow via four tools:

| Tool | Description | Execution |
|------|-------------|-----------|
| `web_search` | Search the web via Serper API | Immediate |
| `fetch_page` | Read a web page via Jina Reader | Immediate |
| `spawn_agents` | Create parallel sub-agents for different research angles | **Deferred** — parent waits for children |
| `write_report` | Produce the final markdown report | Immediate — signals completion |

The key mechanism is **deferred tool resolution**: when an agent calls `spawn_agents`, it can't get results immediately — the children need multiple batch rounds to complete their own research. So the orchestrator pauses the parent, runs the children through the batch loop, and when all children finish, compiles their findings into the tool result. The parent then resumes with full context of what its sub-agents discovered.

```
User: "Research quantum computing"
        │
        ▼
   Root Agent (tools: web_search, fetch_page, spawn_agents, write_report)
        │
        ├─ calls spawn_agents(["error correction", "hardware", "algorithms"])
        │       │
        │       ├─ Sub-agent 0 (error correction)
        │       │     ├─ web_search → fetch_page → web_search → ...
        │       │     ├─ spawn_agents(["surface codes", "topological codes"])
        │       │     │       ├─ Sub-sub-agent → searches, reads, completes
        │       │     │       └─ Sub-sub-agent → searches, reads, completes
        │       │     └─ receives children's findings, writes summary
        │       ├─ Sub-agent 1 → searches, reads, completes
        │       └─ Sub-agent 2 → searches, reads, completes
        │
        ├─ receives all children's findings as tool result
        ├─ optionally searches to fill gaps
        └─ calls write_report("# Final Report\n...")
```

### Orchestrator Loop

All agents at all depths are batched together. A single batch can contain the root's children alongside grandchildren from a different branch:

```
while root not done:
    ┌──────────────────┐
    │  Submit all ready │◄──── parents unblocked by resolved children
    │  agents in batch  │◄──── new children from spawn_agents
    └────────┬─────────┘
             │
        Poll until complete
             │
        Process responses:
        ├─ stop → agent completed, store findings
        ├─ tool_calls:
        │   ├─ web_search/fetch_page → execute immediately
        │   ├─ spawn_agents → create children, pause parent
        │   └─ write_report → store report, mark root done
        └─ length → agent failed
             │
        Resolve waiting parents:
        └─ all children done? → compile findings → unblock parent
             │
        Next iteration
```

### Agent States

```
pending → in_progress → waiting_for_children → in_progress → completed
              │                                                   │
              └─────────────────── failed ◄───────────────────────┘
```

## JSONL Format

The root agent's initial batch request includes all 4 tools:

```json
{
    "custom_id": "root-0-iter-0",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        "messages": [
            {"role": "system", "content": "You are a lead research agent..."},
            {"role": "user", "content": "Research the following topic: quantum computing"}
        ],
        "tools": [
            {"type": "function", "function": {"name": "web_search", ...}},
            {"type": "function", "function": {"name": "fetch_page", ...}},
            {"type": "function", "function": {"name": "spawn_agents", ...}},
            {"type": "function", "function": {"name": "write_report", ...}}
        ],
        "temperature": 0
    }
}
```

Sub-agents get the same tools minus `write_report`. After the root calls `spawn_agents` and children return, the root's messages include:

```json
{"role": "assistant", "tool_calls": [{"id": "call_abc", "function": {"name": "spawn_agents", "arguments": "{\"queries\": [\"error correction\", \"hardware\"]}"}}]},
{"role": "tool", "tool_call_id": "call_abc", "content": "{\"sub_agent_results\": [{\"agent_id\": \"sub-1\", \"findings\": \"...\"}, ...]}"}
```

## Running It

```bash
cd async-agents && uv sync
export DOUBLEWORD_API_KEY="your-key"
export SERPER_API_KEY="your-serper-key"  # Free at https://serper.dev
```

Run a research investigation:

```bash
# Let the root agent decide how to research the topic
uv run async-agents run --topic "quantum computing error correction" -m 235b

# Limit batch rounds for a quicker run
uv run async-agents run --topic "rust vs go for web services" -m 30b --max-iterations 10

# Dry run to inspect the root agent's batch file and tool definitions
uv run async-agents run --topic "nuclear fusion progress" --dry-run
```

Check status of a running batch:

```bash
uv run async-agents status --batch-id <batch-id>
```

View completed reports:

```bash
uv run async-agents report
```

## Architecture

```
src/
├── cli.py      # CLI and single orchestrator loop
├── agent.py    # Agent dataclass, AgentRegistry, tree orchestration
├── tools.py    # Tool definitions (4 tools) and execution dispatch
├── batch.py    # Batch API utilities (JSONL, upload, poll, download)
├── prompts.py  # ROOT_AGENT_SYSTEM and SUB_AGENT_SYSTEM prompts
├── search.py   # Serper API wrapper (called by tools.py)
└── scrape.py   # Jina Reader wrapper (called by tools.py)
```

`agent.py` is the core — it defines the `Agent` dataclass with parent/child relationships and the `AgentRegistry` that manages the tree. The orchestrator functions (`process_responses`, `execute_pending_tools`, `resolve_waiting_parents`) handle the batch loop mechanics. `cli.py` ties it together in a single while loop that runs until the root completes.

## Limitations

- Each batch round adds latency (typically 1-5 minutes depending on queue depth), and recursive spawning multiplies this — a 3-level deep tree might take 10+ batch rounds
- The model may over-delegate (spawning sub-agents for trivially simple queries) or under-delegate (doing everything itself). Prompt engineering helps but isn't perfect
- The Serper API free tier provides 2,500 queries/month. A recursive research run with 5+ agents making 3 searches each can use 20-40 queries per run
- Jina Reader occasionally fails on JavaScript-heavy pages or paywalls
- Context length is a practical limit — agents with many tool call rounds accumulate large message histories
