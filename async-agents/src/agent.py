"""Agent orchestrator for parallel tool-calling research agents.

Each agent independently researches a sub-query using tools (web_search,
fetch_page). All agents are batched together and run in parallel. The
orchestrator loops: submit batch -> poll -> process responses -> execute
tool calls locally -> resubmit agents that need more work.

Agents complete independently — some finish in 2 iterations, others in 5.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import click

from .batch import extract_message, get_finish_reason
from .prompts import RESEARCH_AGENT_SYSTEM
from .tools import TOOL_DEFINITIONS, execute_tool


@dataclass
class Agent:
    """A single research agent with tool-calling capability."""

    id: str
    sub_query: str
    model: str
    status: str = "pending"  # pending | in_progress | completed | failed
    messages: list[dict] = field(default_factory=list)
    iteration: int = 0
    findings: str = ""

    def __post_init__(self):
        if not self.messages:
            self.messages = [
                {"role": "system", "content": RESEARCH_AGENT_SYSTEM},
                {
                    "role": "user",
                    "content": f"Research the following topic thoroughly: {self.sub_query}",
                },
            ]


def create_agents(
    sub_queries: list[str],
    model: str,
) -> list[Agent]:
    """Create one agent per sub-query."""
    return [
        Agent(id=f"agent-{i}", sub_query=q, model=model)
        for i, q in enumerate(sub_queries)
    ]


def build_batch_requests(agents: list[Agent]) -> list[dict]:
    """Build JSONL-ready batch requests for all ready agents."""
    requests_data = []
    for agent in agents:
        if agent.status not in ("pending", "in_progress"):
            continue
        requests_data.append(
            {
                "custom_id": f"{agent.id}-iter-{agent.iteration}",
                "model": agent.model,
                "messages": agent.messages,
                "tools": TOOL_DEFINITIONS,
                "temperature": 0,
                "max_tokens": 4096,
            }
        )
        agent.status = "in_progress"
    return requests_data


def process_responses(agents: list[Agent], results: dict[str, dict]) -> None:
    """Update agent states based on batch results.

    For each agent that was in the batch:
    - finish_reason="stop" -> agent completed, store findings
    - finish_reason="tool_calls" -> store tool calls for execution
    - finish_reason="length" -> agent failed (context too long)
    """
    agent_map = {a.id: a for a in agents}

    for custom_id, result in results.items():
        # custom_id format: "agent-{id}-iter-{n}"
        parts = custom_id.rsplit("-iter-", 1)
        agent_id = parts[0]
        agent = agent_map.get(agent_id)
        if not agent:
            continue

        finish_reason = get_finish_reason(result)
        message = extract_message(result)

        if finish_reason == "stop":
            # Agent decided it's done — append final message and mark complete
            agent.messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content", ""),
                }
            )
            agent.findings = message.get("content", "")
            agent.status = "completed"
            agent.iteration += 1

        elif finish_reason in ("tool_calls", "function_call"):
            # Agent wants to call tools — append the assistant message with tool_calls
            assistant_msg = {"role": "assistant", "content": message.get("content")}
            if "tool_calls" in message:
                assistant_msg["tool_calls"] = message["tool_calls"]
            agent.messages.append(assistant_msg)
            agent.iteration += 1
            # Status stays in_progress — tools need execution

        else:
            # length, error, etc.
            agent.status = "failed"
            agent.iteration += 1


def execute_pending_tools(agents: list[Agent], max_workers: int = 10) -> None:
    """Execute pending tool calls for all agents that have them.

    Tool calls are executed in parallel across all agents. Results are
    appended to each agent's message history as role="tool" messages.
    """
    # Collect all tool calls with their agent references
    work_items = []  # (agent, tool_call)
    for agent in agents:
        if agent.status != "in_progress":
            continue
        # Check if the last message has tool_calls
        if not agent.messages:
            continue
        last_msg = agent.messages[-1]
        if last_msg.get("role") != "assistant" or "tool_calls" not in last_msg:
            continue
        for tc in last_msg["tool_calls"]:
            work_items.append((agent, tc))

    if not work_items:
        return

    click.echo(
        f"  Executing {len(work_items)} tool calls across {_count_agents_with_tools(agents)} agents..."
    )

    # Execute all tool calls in parallel
    results_map: dict[str, tuple[Agent, str]] = {}  # tool_call_id -> (agent, result)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for agent, tc in work_items:
            func = tc["function"]
            future = executor.submit(execute_tool, func["name"], func["arguments"])
            futures[future] = (agent, tc)

        for future in as_completed(futures):
            agent, tc = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = json.dumps({"error": str(e)})
            results_map[tc["id"]] = (agent, result)

    # Append tool results to agent messages in the correct order
    for agent in agents:
        if agent.status != "in_progress":
            continue
        if not agent.messages:
            continue
        last_msg = agent.messages[-1]
        if last_msg.get("role") != "assistant" or "tool_calls" not in last_msg:
            continue
        for tc in last_msg["tool_calls"]:
            if tc["id"] in results_map:
                _, result_content = results_map[tc["id"]]
                agent.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_content,
                    }
                )


def get_ready_agents(agents: list[Agent]) -> list[Agent]:
    """Get agents that are ready for the next batch submission."""
    ready = []
    for agent in agents:
        if agent.status not in ("pending", "in_progress"):
            continue
        # An in_progress agent is ready if its last message is a tool result
        # (meaning tool execution is done and it needs another model turn)
        if agent.status == "in_progress" and agent.messages:
            last_msg = agent.messages[-1]
            if last_msg.get("role") == "tool":
                ready.append(agent)
            # Also ready if status is pending (first iteration)
        elif agent.status == "pending":
            ready.append(agent)
    return ready


def all_done(agents: list[Agent]) -> bool:
    """Check if all agents have completed or failed."""
    return all(a.status in ("completed", "failed") for a in agents)


def _count_agents_with_tools(agents: list[Agent]) -> int:
    """Count agents that have pending tool calls."""
    count = 0
    for agent in agents:
        if agent.status != "in_progress" or not agent.messages:
            continue
        last_msg = agent.messages[-1]
        if last_msg.get("role") == "assistant" and "tool_calls" in last_msg:
            count += 1
    return count
