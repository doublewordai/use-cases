# Async Agents: Recursive Research via Batch API

A single root agent drives the entire research process — breaking a topic into sub-queries, spawning parallel sub-agents, and synthesizing a final report. Sub-agents can recursively spawn their own sub-agents, creating a tree of autonomous researchers. All agents at all depths run in parallel within each batch round, and the model decides everything: what to read, when to delegate, and when to write the report.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Design Philosophy: Wide Trees via Search-First

This system is optimized for **batch inference** — where the cost of one request is the same whether you submit 1 or 100 in a batch. The key insight: **make the tree as wide as possible** so each batch round does maximum work in parallel.

The mechanism is **search-first agent creation**: when any agent is created (root or sub-agent), a web search is executed immediately and the results are injected into the agent's initial messages. This means:

- **Round 0**: Root agent already has search results → spawns 5-8 sub-agents immediately
- **Round 1**: All sub-agents already have search results → read pages and/or spawn their own sub-agents in parallel
- **Round 2**: Sub-agents complete with findings; grandchildren (if any) are also working

Compare this with a naive approach where each agent wastes its first batch round calling a search tool and waiting for results. That sequential pattern makes trees deep and narrow — the opposite of what batch inference rewards.

**Breadth beats depth**: spawning 5 sub-agents that each complete in 2 rounds is faster than 1 agent doing 10 sequential search-read-search-read cycles, because all 5 sub-agents run in the same batch rounds.

## How It Works

The model controls the research workflow via five tools:

| Tool | Description | Execution |
|------|-------------|-----------|
| `search` | Search the web for a new angle (initial topic is pre-searched) | Immediate |
| `read_pages` | Read web pages via Jina Reader | Immediate |
| `spawn_agents` | Create parallel sub-agents (each gets pre-searched results) | **Deferred** — parent waits |
| `reference_findings` | Retrieve another agent's completed findings | **Deferred** |
| `write_report` | Produce the final markdown report | **Deferred** — signals completion |

### Search-First Agent Lifecycle

Every agent — root or sub-agent — follows this lifecycle:

```
Creation:  web search executed immediately, results injected into messages
    │
    ▼
Round 1:   agent sees search results, calls read_pages + spawn_agents in parallel
    │
    ▼
Round 2:   agent sees page content, writes findings (or waits for children)
```

This contrasts with the old sequential pattern (search → wait → read → wait → act) that took 3-5 rounds per agent.

### Agent Tree Example

```
User: "Research quantum computing"
        │
        ▼
   Root Agent (pre-searched, has results in context)
        │
        ├─ calls spawn_agents(["error correction", "hardware", "algorithms"])
        │       │
        │       │  (all children get pre-searched results at creation)
        │       │
        │       ├─ Sub-agent 0 (error correction, pre-searched)
        │       │     ├─ read_pages([url1, url2, url3])  ← first response
        │       │     └─ writes findings                  ← second response
        │       │
        │       ├─ Sub-agent 1 (hardware, pre-searched)
        │       │     ├─ read_pages + spawn_agents in parallel  ← first response
        │       │     │       ├─ Sub-sub-agent (pre-searched) → reads, completes
        │       │     │       └─ Sub-sub-agent (pre-searched) → reads, completes
        │       │     └─ receives children's findings, writes summary
        │       │
        │       └─ Sub-agent 2 (algorithms, pre-searched)
        │             └─ read_pages → writes findings
        │
        ├─ receives all children's findings as tool result
        └─ calls write_report("# Final Report\n...")
```

### Orchestrator Loop

All agents at all depths are batched together. A single batch can contain the root's children alongside grandchildren from a different branch:

```
while root not done:
    ┌──────────────────┐
    │  Submit all ready │◄──── parents unblocked by resolved children
    │  agents in batch  │◄──── new children (pre-searched at creation)
    └────────┬─────────┘
             │
        Poll until complete
             │
        Process responses:
        ├─ stop → agent completed, store findings
        ├─ tool_calls:
        │   ├─ search/read_pages → execute immediately
        │   ├─ spawn_agents → search children's topics, create children, pause parent
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

The root agent's initial batch request includes pre-searched results and all 5 tools:

```json
{
    "custom_id": "root-0-iter-0",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        "messages": [
            {"role": "system", "content": "You are a lead research agent..."},
            {"role": "user", "content": "Research the following topic: quantum computing"},
            {"role": "system", "content": "Initial search results for your topic:\n\n1. [Title](url)\n   Snippet...\n\n2. ..."}
        ],
        "tools": [
            {"type": "function", "function": {"name": "search", ...}},
            {"type": "function", "function": {"name": "read_pages", ...}},
            {"type": "function", "function": {"name": "spawn_agents", ...}},
            {"type": "function", "function": {"name": "reference_findings", ...}},
            {"type": "function", "function": {"name": "write_report", ...}}
        ],
        "temperature": 0
    }
}
```

Sub-agents get the same tools minus `write_report`, and also include pre-searched results in their initial messages.

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

## Cost

Batch inference is cheap. An example "effects of modern lifestyle on health" query using the Qwen 235B model:

| Metric | Value |
|--------|-------|
| Agents | 66 |
| Batch rounds | 21 |
| Input tokens | ~3.0M |
| Output tokens | ~113K |
| **Total cost** | **$0.34** |

A smaller run with the 30B model (8-15 agents, 5-10 rounds) typically costs under $0.05. The search-first approach reduces rounds per agent, which keeps input token accumulation (and therefore cost) low.

## Architecture

```
src/
├── cli.py                  # CLI and single orchestrator loop
├── prompts.py              # ROOT_AGENT_SYSTEM and SUB_AGENT_SYSTEM prompts
├── tools/
│   ├── __init__.py         # Tool definitions (5 tools) and execution dispatch
│   ├── search.py           # Serper API wrapper (called by tools and orchestrator)
│   └── scrape.py           # Jina Reader wrapper (called by tools)
└── utils/
    ├── batch.py            # Batch API utilities (JSONL, upload, poll, download)
    └── agent.py            # Agent dataclass, AgentRegistry, tree orchestration
```

`utils/agent.py` is the core — it defines the `Agent` dataclass with parent/child relationships and the `AgentRegistry` that manages the tree. The search-first logic lives in `create_root` and `spawn_children`, which execute web searches at agent creation time. The orchestrator functions (`process_responses`, `execute_pending_tools`, `resolve_waiting_parents`) handle the batch loop mechanics. `cli.py` ties it together in a single while loop that runs until the root completes.

## Limitations

- Each batch round adds latency (typically 1-5 minutes depending on queue depth), but the search-first approach minimizes the number of rounds needed
- The model may over-delegate (spawning sub-agents for trivially simple queries) or under-delegate (doing everything itself). Prompt engineering helps but isn't perfect
- The Serper API free tier provides 2,500 queries/month. Search-first uses one search per agent at creation time, plus any additional searches the model requests. A run with 10 agents uses ~10-15 searches
- Jina Reader occasionally fails on JavaScript-heavy pages or paywalls
- Context length is a practical limit — agents with many tool call rounds accumulate large message histories, though the search-first approach reduces rounds per agent
