# Async Agents: Parallel Tool-Calling Research via Batch API

Multiple research agents run in parallel, each independently deciding what to search and read using tool calling. All agents are batched together — the orchestrator loops: submit batch, poll for results, execute tool calls locally, resubmit agents that need more work. Agents complete independently; some finish in 2 iterations, others in 5.

A 5-agent research task that would cost ~$2.80 in realtime GPT-5.2 calls costs $0.14 with Doubleword's batch Qwen3-235B.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## How It Works

The key difference from a typical agent: **the model decides what to do via tool calling**, rather than the code hardcoding a search→fetch→analyze pipeline. Each agent receives tools (`web_search`, `fetch_page`) and autonomously decides:
- What queries to search
- Which results to read in detail
- When to search for more information
- When it has enough to write its findings

```
User provides topic
        │
        ▼
Generate N sub-queries (one batch)
        │
        ▼
Create N agents, one per sub-query
Each has: system prompt, tools, user message
        │
        ▼
┌──────────────────┐
│   BATCH LOOP     │◄──────────────────────────┐
│                  │                            │
│ Submit all ready │    Agents with tool        │
│ agents in one    │    results resubmit        │
│ batch            │                            │
└────────┬─────────┘                            │
         │                                      │
    Poll until complete                         │
         │                                      │
    Process responses:                          │
    ├─ stop → agent done, store findings        │
    ├─ tool_calls → execute tools locally ──────┘
    └─ length → agent failed
         │
         ▼
All agents done → Synthesis batch → Report
```

Within each batch round, all active agents are submitted together. The model returns either:
- **Tool calls** — the orchestrator executes them locally (web search via Serper, page fetch via Jina Reader), appends results, and resubmits the agent in the next batch
- **A final text response** — the agent is done, its findings are stored for synthesis

## JSONL Format with Tools

Each line in the batch file includes tool definitions:

```json
{
    "custom_id": "agent-0-iter-0",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        "messages": [
            {"role": "system", "content": "You are a research agent..."},
            {"role": "user", "content": "Research: quantum error correction advances"}
        ],
        "tools": [
            {"type": "function", "function": {"name": "web_search", ...}},
            {"type": "function", "function": {"name": "fetch_page", ...}}
        ],
        "temperature": 0
    }
}
```

After tool execution, the agent's conversation grows with tool call/result pairs:

```json
"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Research: quantum error correction"},
    {"role": "assistant", "tool_calls": [{"id": "call_abc", "function": {"name": "web_search", "arguments": "{\"query\": \"quantum error correction 2025\"}"}}]},
    {"role": "tool", "tool_call_id": "call_abc", "content": "{\"results\": [...]}"},
    {"role": "assistant", "tool_calls": [{"id": "call_def", "function": {"name": "fetch_page", "arguments": "{\"url\": \"https://...\"}"}}]},
    {"role": "tool", "tool_call_id": "call_def", "content": "Page content..."},
    {"role": "assistant", "content": "Based on my research, here are the key findings..."}
]
```

## Running It

```bash
cd async-agents && uv sync
export DOUBLEWORD_API_KEY="your-key"
export SERPER_API_KEY="your-serper-key"  # Free at https://serper.dev
```

Run a research investigation:

```bash
# 5 parallel agents, each using tool calling to research independently
uv run async-agents run --topic "quantum computing error correction" -m 235b --agents 5

# Fewer agents for a quick test
uv run async-agents run --topic "rust vs go for web services" -m 30b --agents 3 --max-iterations 5

# Dry run to inspect the batch file and tool definitions
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

5 parallel research agents, each making 3-5 tool calls:

| Provider | Model | Mode | Avg Cost |
|----------|-------|------|----------|
| OpenAI | GPT-5.2 | Realtime | ~$2.80 |
| OpenAI | GPT-5.2 | Batch | ~$1.40 |
| Doubleword | Qwen3-235B | Batch (24h) | ~$0.14 |
| Doubleword | Qwen3-30B | Batch (24h) | ~$0.05 |

## Architecture

```
src/
├── cli.py      # CLI commands and orchestrator loop
├── agent.py    # Agent dataclass and orchestrator functions
├── tools.py    # Tool definitions (JSON Schema) and local execution
├── batch.py    # Batch API utilities (JSONL creation, upload, polling)
├── prompts.py  # System prompts for research agents and synthesis
├── search.py   # Serper API wrapper (called by tools.py)
└── scrape.py   # Jina Reader wrapper (called by tools.py)
```

The orchestrator pattern: `agent.py` defines the Agent class and functions for building batch requests, processing responses, and executing tool calls. `cli.py` ties it together in a loop. `tools.py` defines the tool schemas the model receives and dispatches execution to `search.py` and `scrape.py`.

## Limitations

- Tool call execution happens locally between batch rounds, so each round of tool calls adds batch latency (typically 1-5 minutes per round depending on queue depth)
- Agents may use their iteration budget inefficiently — searching for the same thing twice or reading low-value pages. A more sophisticated orchestrator could prune redundant calls
- The Serper API free tier provides 2,500 queries/month. With 5 agents making ~3 searches each, one research run uses ~15 queries
- Jina Reader occasionally fails on JavaScript-heavy pages or paywalls
