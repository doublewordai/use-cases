# Async Agents: Deep Research in a Day for \$0.34

Agentic workflows require many rounds of inference. A single research agent might call tools 10-20 times before producing a final report; a tree of agents can easily require hundreds of LLM calls. At real-time API rates, this gets expensive fast. At batch rates with a 24-hour SLA, it's cheap but slow: each batch round adds latency, and a deep agent tree might take weeks to complete.

Doubleword is the only platform that offers a 1-hour SLA for batch inference. This changes what's possible: a recursive research system that spawns dozens of sub-agents, each making multiple tool calls, can complete in a day rather than a month, while still costing 95% less than real-time inference.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## Why This Matters

Consider a research query like "how has the popularity of daffodils evolved over recent centuries?" A system that spawns sub-agents to explore different angles (cultural history, agricultural data, literary references, regional variations) might require 20+ batch rounds with 30-60 agents total. With a 24-hour SLA, that's potentially a month of wall-clock time. With Doubleword's 1-hour SLA, the same research completes in a day.

The cost difference is equally dramatic. Here's what that daffodils query actually cost using the Qwen 235B model (1,950,090 input tokens, 93,031 output tokens):

| Provider | Model | ELO | Input Rate | Output Rate | Total Cost |
|----------|-------|-----|------------|-------------|------------|
| Doubleword (1hr SLA) | Qwen 235B | 1423 | \$0.15/MTok | \$0.55/MTok | **\$0.34** |
| OpenAI | GPT-4o | 1442 | \$2.50/MTok | \$10.00/MTok | **\$5.81** |
| Anthropic | Claude Sonnet 4.5 | 1450 | \$3.00/MTok | \$15.00/MTok | **\$7.25** |

Qwen 235B scores 1423 on the [LMArena leaderboard](https://kearai.com/leaderboard/chat), comparable to GPT-4o (1442) and Claude Sonnet 4.5 (1450). The capability is similar; the cost is 16-20x lower.

ELO scores from [KEAR AI Chatbot Arena](https://kearai.com/leaderboard/chat) (January 2026). Pricing from [OpenAI](https://openai.com/api/pricing/) and [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing).

## How It Works

A single root agent drives the entire research process: breaking a topic into sub-queries, spawning parallel sub-agents, and synthesizing a final report. Sub-agents can recursively spawn their own sub-agents, creating a tree of autonomous researchers. All agents at all depths run in parallel within each batch round, and the model decides everything: what to read, when to delegate, and when to write the report.

The system uses [Serper](https://serper.dev) to retrieve web search results and [Jina Reader](https://jina.ai/reader/) to fetch web pages as clean markdown that the model can process.

### Design Philosophy: Wide Trees via Search-First

This system is optimized for batch inference, where the cost of inference is low but time delays can be significant. The key insight: make the tree as wide as possible so each batch round does maximum work in parallel.

The mechanism is search-first agent creation: when any agent is created (root or sub-agent), a web search is executed immediately and the results are injected into the agent's initial messages. This means:

- **Round 0**: Root agent already has search results, spawns 5-8 sub-agents immediately
- **Round 1**: All sub-agents already have search results, read pages and/or spawn their own sub-agents in parallel
- **Round N**: Sub-agents complete with findings

Compare this with a naive approach where each agent wastes its first batch round calling a search tool and waiting for results. That sequential pattern makes trees deep and narrow, the opposite of what batch inference rewards.

Breadth beats depth: spawning 5 sub-agents that each complete in 2 rounds is faster than 1 agent doing 10 sequential search-read-search-read cycles, because all 5 sub-agents run in the same batch rounds.

### Tools

The model controls the research workflow via five tools:

| Tool | Description | Execution |
|------|-------------|-----------|
| `search` | Search the web for a new angle (initial topic is pre-searched) | Immediate |
| `read_pages` | Read web pages via Jina Reader | Immediate |
| `spawn_agents` | Create parallel sub-agents (each gets pre-searched results) | **Deferred** (parent waits) |
| `reference_findings` | Retrieve another agent's completed findings | **Deferred** |
| `write_report` | Produce the final markdown report | **Deferred** (signals completion) |

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

## Cost Comparison

Here's what actual research runs cost with the 235B model:

| Query | Agents | Rounds | Input Tokens | Output Tokens | Doubleword (1hr) | OpenAI GPT-4o | Claude Sonnet 4.5 |
|-------|--------|--------|--------------|---------------|------------------|---------------|-------------------|
| Daffodil popularity history | 47 | 25 | 1.95M | 93K | **\$0.34** | \$5.81 | \$7.25 |
| Effects of modern lifestyle on health | 66 | 21 | 3.0M | 113K | **\$0.51** | \$8.63 | \$10.70 |

The search-first approach reduces rounds per agent, which keeps input token accumulation (and therefore cost) low. A smaller run with the 30B model (8-15 agents, 5-10 rounds) typically costs under \$0.05.

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

`utils/agent.py` is the core. It defines the `Agent` dataclass with parent/child relationships and the `AgentRegistry` that manages the tree. The search-first logic lives in `create_root` and `spawn_children`, which execute web searches at agent creation time. The orchestrator functions (`process_responses`, `execute_pending_tools`, `resolve_waiting_parents`) handle the batch loop mechanics. `cli.py` ties it together in a single while loop that runs until the root completes.

## Limitations

- Each batch round adds latency (typically 5-10 minutes depending on queue depth), but the search-first approach minimizes the number of rounds needed
- The model may over-delegate (spawning sub-agents for trivially simple queries) or under-delegate (doing everything itself). Prompt engineering helps but isn't perfect
- The Serper API free tier provides 2,500 queries/month. Search-first uses one search per agent at creation time, plus any additional searches the model requests. A run with 10 agents uses around 10-15 searches
- Jina Reader occasionally fails on JavaScript-heavy pages or paywalls
- Context length is a practical limit: agents with many tool call rounds accumulate large message histories, though the search-first approach reduces rounds per agent
