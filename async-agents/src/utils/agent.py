"""Recursive agent tree with deferred tool resolution.

The root agent can spawn sub-agents via the spawn_agents tool. Sub-agents
can spawn their own sub-agents, creating an arbitrarily deep tree. All
agents at all depths are batched together and run in parallel.

spawn_agents is a deferred tool — when a parent calls it, the parent is
paused (waiting_for_children) until all children complete. Children's
findings are then compiled into the tool result and the parent resumes.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import click

from ..prompts import ROOT_AGENT_SYSTEM, SUB_AGENT_SYSTEM
from ..tools import execute_tool, get_tools_for_agent
from ..tools.search import format_results_for_context, search
from .batch import extract_message, get_finish_reason


@dataclass
class PendingSpawn:
    """Tracks a single spawn_agents tool call waiting for children."""

    tool_call_id: str
    children_ids: list[str] = field(default_factory=list)


@dataclass
class Agent:
    """A research agent in the recursive tree."""

    id: str
    model: str
    status: str = "pending"  # pending|in_progress|waiting_for_children|completed|failed
    messages: list[dict] = field(default_factory=list)
    iteration: int = 0
    findings: str = ""
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    pending_spawns: list[PendingSpawn] = field(default_factory=list)
    depth: int = 0
    is_root: bool = False
    report: str = ""  # Only set on root when write_report is called
    last_tool: str = ""  # Last tool call name for tree display
    verified_urls: list[dict] = field(default_factory=list)  # [{url, title}]


class AgentRegistry:
    """Manages the tree of agents and provides orchestrator helpers."""

    def __init__(self, max_depth: int = 3, max_agent_iterations: int = 8):
        self.agents: dict[str, Agent] = {}
        self.max_depth = max_depth
        self.max_agent_iterations = max_agent_iterations
        self._next_id = 0

    def _gen_id(self, prefix: str = "agent") -> str:
        id_ = f"{prefix}-{self._next_id}"
        self._next_id += 1
        return id_

    def create_root(self, topic: str, model: str) -> Agent:
        """Create the root agent for a research topic.

        Immediately executes a web search for the topic and injects the
        results into the agent's messages so the first batch inference
        call already has data to act on.
        """
        messages = [
            {"role": "system", "content": ROOT_AGENT_SYSTEM},
            {
                "role": "user",
                "content": f"Research the following topic and produce a comprehensive report: {topic}",
            },
        ]

        # Search-first: execute search at creation time
        try:
            results = search(topic)
            formatted = format_results_for_context(topic, results)
            messages.append(
                {
                    "role": "system",
                    "content": f"Initial search results for your topic:\n\n{formatted}",
                }
            )
        except Exception as e:
            click.echo(f"  Pre-search failed: {e}")
            messages.append(
                {
                    "role": "system",
                    "content": f"Initial search failed ({e}). Use the search tool to find information.",
                }
            )

        agent = Agent(
            id=self._gen_id("root"),
            model=model,
            is_root=True,
            depth=0,
            messages=messages,
        )
        self.agents[agent.id] = agent
        return agent

    def spawn_children(self, parent: Agent, queries: list[str]) -> list[Agent]:
        """Create child agents for the given parent.

        Immediately executes web searches for all children in parallel
        and injects results into their messages so each child's first
        batch inference call already has data to act on.
        """
        # Search-first: execute all searches in parallel at creation time
        search_results: dict[str, dict | None] = {}
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = {executor.submit(search, query): query for query in queries}
            for future in as_completed(futures):
                query = futures[future]
                try:
                    search_results[query] = future.result()
                except Exception as e:
                    click.echo(f"  Pre-search failed for {query!r}: {e}")
                    search_results[query] = None

        children = []
        for query in queries:
            # Skip agents whose pre-search failed
            results = search_results.get(query)
            if results is None:
                continue

            formatted = format_results_for_context(query, results)
            messages = [
                {"role": "system", "content": SUB_AGENT_SYSTEM},
                {
                    "role": "user",
                    "content": f"Research the following topic thoroughly: {query}",
                },
                {
                    "role": "system",
                    "content": f"Initial search results for your topic:\n\n{formatted}",
                },
            ]

            child = Agent(
                id=self._gen_id("sub"),
                model=parent.model,
                parent_id=parent.id,
                depth=parent.depth + 1,
                messages=messages,
            )
            self.agents[child.id] = child
            children.append(child)
            parent.children_ids.append(child.id)
        return children

    def get_root(self) -> Agent:
        """Get the root agent."""
        for agent in self.agents.values():
            if agent.is_root:
                return agent
        raise RuntimeError("No root agent found")

    def root_done(self) -> bool:
        """Check if the root agent has completed or failed."""
        root = self.get_root()
        return root.status in ("completed", "failed")

    def get_ready_agents(self) -> list[Agent]:
        """Get agents ready for the next batch submission.

        An agent is ready if:
        - status is 'pending' (first submission), OR
        - status is 'in_progress' and its last message is role='tool'
          (tool results appended, ready for next model turn)

        Agents that have exceeded max_agent_iterations complete with
        whatever findings they have so far.
        """
        ready = []
        for agent in self.agents.values():
            if agent.status == "pending":
                ready.append(agent)
            elif agent.status == "in_progress" and agent.messages:
                if agent.messages[-1].get("role") == "tool":
                    if agent.iteration >= self.max_agent_iterations:
                        agent.status = "completed"
                        agent.findings = agent.findings or "Max iterations reached."
                    else:
                        ready.append(agent)
        return ready

    def _build_agent_context(self, for_agent: Agent) -> str:
        """Build a context string listing all other agents and their status.

        Injected into each request so the model can see what topics are
        already being researched and use reference_findings to avoid
        redundant work.
        """
        lines = ["Other agents in this research session:"]
        for agent in self.agents.values():
            if agent.id == for_agent.id:
                continue
            query = agent.messages[1]["content"] if len(agent.messages) > 1 else ""
            for prefix in [
                "Research the following topic and produce a comprehensive report: ",
                "Research the following topic thoroughly: ",
            ]:
                if query.startswith(prefix):
                    query = query[len(prefix) :]
                    break
            has_findings = "yes" if agent.findings else "no"
            lines.append(
                f"  - {agent.id} [{agent.status}] (findings: {has_findings}): {query[:80]}"
            )
        if len(lines) == 1:
            return ""  # No other agents yet
        lines.append("")
        lines.append(
            "Use reference_findings(agent_id) to reuse another agent's "
            "research instead of re-searching the same topic."
        )
        return "\n".join(lines)

    def build_batch_requests(self, agents: list[Agent]) -> list[dict]:
        """Build JSONL-ready batch requests for the given agents."""
        requests_data = []
        for agent in agents:
            tools = get_tools_for_agent(agent.is_root)
            messages = list(agent.messages)

            # Inject agent context so the model knows about sibling agents
            context = self._build_agent_context(agent)
            if context:
                messages = messages + [{"role": "system", "content": context}]

            requests_data.append(
                {
                    "custom_id": f"{agent.id}-iter-{agent.iteration}",
                    "model": agent.model,
                    "messages": messages,
                    "tools": tools,
                    "temperature": 0,
                    "max_tokens": 8192 if agent.is_root else 4096,
                }
            )
            agent.status = "in_progress"
        return requests_data

    def get_children(self, parent: Agent) -> list[Agent]:
        """Get all children of a parent agent."""
        return [self.agents[cid] for cid in parent.children_ids if cid in self.agents]

    def collect_verified_urls(self, agent: Agent) -> list[dict]:
        """Collect all verified URLs from an agent and all its descendants."""
        seen = set()
        urls = []
        stack = [agent]
        while stack:
            a = stack.pop()
            for entry in a.verified_urls:
                if entry["url"] not in seen:
                    seen.add(entry["url"])
                    urls.append(entry)
            for cid in a.children_ids:
                if cid in self.agents:
                    stack.append(self.agents[cid])
        return urls

    def agent_count(self) -> dict[str, int]:
        """Count agents by status."""
        counts: dict[str, int] = {}
        for agent in self.agents.values():
            counts[agent.status] = counts.get(agent.status, 0) + 1
        return counts

    def print_tree(self, iteration: int | None = None) -> None:
        """Print a compact status header and the full agent tree."""
        STATUS_ICONS = {
            "pending": "○",
            "in_progress": "◉",
            "waiting_for_children": "◎",
            "completed": "●",
            "failed": "✗",
        }

        TOPIC_PREFIXES = [
            "Research the following topic and produce a comprehensive report: ",
            "Research the following topic thoroughly: ",
            "Research the following topic: ",
        ]

        def _extract_label(agent: Agent, max_len: int = 50) -> str:
            if len(agent.messages) < 2:
                return agent.id
            text = agent.messages[1]["content"]
            for prefix in TOPIC_PREFIXES:
                if text.startswith(prefix):
                    text = text[len(prefix) :]
                    break
            return text[:max_len]

        def _tool_suffix(agent: Agent) -> str:
            if agent.last_tool:
                return f"  [{agent.last_tool}]"
            if agent.status == "waiting_for_children":
                return "  [waiting...]"
            return ""

        def _print_agent(agent: Agent, prefix: str = "", is_last: bool = True):
            connector = "└─ " if is_last else "├─ "
            icon = STATUS_ICONS.get(agent.status, "?")
            label = _extract_label(agent)
            click.echo(f"  {prefix}{connector}{icon} {label}{_tool_suffix(agent)}")
            child_prefix = prefix + ("   " if is_last else "│  ")
            children = [
                self.agents[cid] for cid in agent.children_ids if cid in self.agents
            ]
            for i, child in enumerate(children):
                _print_agent(child, child_prefix, i == len(children) - 1)

        # Header
        counts = self.agent_count()
        total = len(self.agents)
        parts = []
        if iteration is not None:
            parts.append(f"Round {iteration}")
        parts.append(f"{total} agents")
        for status in (
            "in_progress",
            "pending",
            "waiting_for_children",
            "completed",
            "failed",
        ):
            n = counts.get(status, 0)
            if n > 0:
                short = {"in_progress": "running", "waiting_for_children": "waiting"}
                parts.append(f"{n} {short.get(status, status)}")
        click.echo(f"── {' · '.join(parts)} ──")

        # Tree
        root = self.get_root()
        icon = STATUS_ICONS.get(root.status, "?")
        label = _extract_label(root, max_len=60)
        click.echo(f"  {icon} {label}{_tool_suffix(root)}")
        children = [self.agents[cid] for cid in root.children_ids if cid in self.agents]
        for i, child in enumerate(children):
            _print_agent(child, "", i == len(children) - 1)

    def build_final_report_request(self, root: Agent) -> list[dict]:
        """Build a tool-free batch request for the root agent's final round.

        Instead of giving the model write_report as a tool (which would
        require two inference calls), we send no tools and a user message
        asking for the report directly. The response content IS the report.
        """
        messages = list(root.messages)

        # Inject agent context so the model sees all findings
        context = self._build_agent_context(root)
        if context:
            messages.append({"role": "system", "content": context})

        # Collect all verified URLs from the entire tree
        all_urls = self.collect_verified_urls(root)
        sources_block = ""
        if all_urls:
            source_lines = [f"- [{e['title']}]({e['url']})" for e in all_urls]
            sources_block = (
                "\n\nVERIFIED SOURCES — these URLs were actually fetched and "
                "read during research. Use ONLY these for citations:\n"
                + "\n".join(source_lines)
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    "All research is now complete. Based on all the findings "
                    "above, write a comprehensive, well-structured research "
                    "report in markdown. Include an executive summary, "
                    "thematic sections with source citations, areas where "
                    "sources disagree, and areas for further research.\n\n"
                    "IMPORTANT: Only cite URLs from the verified sources list "
                    "below. Do not cite URLs from search snippets or invent "
                    "URLs. If a finding has no verified URL, state it without "
                    "a link.\n\n"
                    "Output ONLY the report — no preamble or commentary."
                    + sources_block
                ),
            }
        )

        root.status = "in_progress"
        return [
            {
                "custom_id": f"{root.id}-iter-{root.iteration}-final",
                "model": root.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 8192,
            }
        ]

    def force_complete_all(self) -> None:
        """Force-complete all unfinished non-root agents and resolve the tree.

        Called when the global iteration limit is hit. Completes agents
        bottom-up so parent resolution cascades correctly. The root agent
        is left in in_progress so it can get a final batch round to
        write its report with whatever findings were collected.
        """
        root = self.get_root()

        # Sort by depth descending so leaves complete before parents
        by_depth = sorted(self.agents.values(), key=lambda a: a.depth, reverse=True)

        for agent in by_depth:
            if agent.status in ("completed", "failed"):
                continue
            if agent.is_root:
                continue  # Don't complete root — it needs a final round
            agent.status = "completed"
            if not agent.findings:
                agent.findings = "Research incomplete — global iteration limit reached."

        # Resolve waiting parents up the tree so root gets unblocked
        for _ in range(self.max_depth + 2):
            resolve_waiting_parents(self)
            # Stop once root is unblocked (in_progress with tool results)
            if root.status == "in_progress":
                break


def process_responses(registry: AgentRegistry, results: dict[str, dict]) -> None:
    """Update agent states based on batch results.

    - finish_reason='stop' → agent completed, store findings
    - finish_reason='tool_calls' → append assistant message with tool_calls
    - finish_reason='length' or other → agent failed
    """
    for custom_id, result in results.items():
        # custom_id format: "{agent_id}-iter-{n}"
        agent_id = custom_id.rsplit("-iter-", 1)[0]
        agent = registry.agents.get(agent_id)
        if not agent:
            continue

        finish_reason = get_finish_reason(result)
        message = extract_message(result)

        if finish_reason == "stop":
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
            assistant_msg = {"role": "assistant", "content": message.get("content")}
            if "tool_calls" in message:
                assistant_msg["tool_calls"] = message["tool_calls"]
            agent.messages.append(assistant_msg)
            agent.iteration += 1
            # Status stays in_progress — tools need execution

        else:
            agent.status = "failed"
            agent.iteration += 1


def execute_pending_tools(
    registry: AgentRegistry, max_workers: int = 10
) -> dict[str, int]:
    """Execute pending tool calls for all agents.

    Immediate tools (search, read_pages) are executed in parallel.
    Deferred tools are handled specially:
    - spawn_agents: creates children, sets parent to waiting_for_children
    - write_report: stores report, marks agent completed

    Returns:
        Dict mapping tool name -> call count for this round.
    """
    tool_counts: dict[str, int] = {}

    # Collect agents that have tool_calls in their last message
    agents_with_tools = []
    for agent in registry.agents.values():
        if agent.status != "in_progress" or not agent.messages:
            continue
        last_msg = agent.messages[-1]
        if last_msg.get("role") == "assistant" and "tool_calls" in last_msg:
            agents_with_tools.append(agent)

    if not agents_with_tools:
        return tool_counts

    # Separate immediate and deferred tool calls
    immediate_work = []  # (agent, tool_call)
    deferred_work = []  # (agent, tool_call)

    for agent in agents_with_tools:
        tool_names = []
        for tc in agent.messages[-1]["tool_calls"]:
            name = tc["function"]["name"]
            tool_names.append(name)
            tool_counts[name] = tool_counts.get(name, 0) + 1
            if name in ("spawn_agents", "write_report", "reference_findings"):
                deferred_work.append((agent, tc))
            else:
                immediate_work.append((agent, tc))
        agent.last_tool = ", ".join(tool_names)

    # Execute immediate tools in parallel
    if immediate_work:
        # tool_call_id -> (result_str, tool_name)
        immediate_results: dict[str, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for agent, tc in immediate_work:
                func = tc["function"]
                future = executor.submit(execute_tool, func["name"], func["arguments"])
                futures[future] = tc

            for future in as_completed(futures):
                tc = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = json.dumps({"error": str(e)})
                immediate_results[tc["id"]] = (result, tc["function"]["name"])

        # Append immediate tool results to agent messages and track verified URLs
        for agent in agents_with_tools:
            last_msg = agent.messages[-1]
            if last_msg.get("role") != "assistant" or "tool_calls" not in last_msg:
                continue
            for tc in last_msg["tool_calls"]:
                if tc["id"] in immediate_results:
                    result_str, tool_name = immediate_results[tc["id"]]
                    agent.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_str,
                        }
                    )
                    # Track successfully fetched URLs
                    if tool_name == "read_pages":
                        try:
                            parsed = json.loads(result_str)
                            for page in parsed.get("pages", []):
                                if "url" in page and "error" not in page:
                                    title = page.get("content", "")[:100].split("\n")[0]
                                    agent.verified_urls.append(
                                        {"url": page["url"], "title": title}
                                    )
                        except (json.JSONDecodeError, TypeError):
                            pass

    # Handle deferred tools
    for agent, tc in deferred_work:
        name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])

        if name == "spawn_agents":
            queries = args.get("queries", [])
            if not queries:
                agent.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": "No queries provided"}),
                    }
                )
                continue

            # Enforce max depth
            if agent.depth >= registry.max_depth:
                agent.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(
                            {
                                "error": f"Maximum depth ({registry.max_depth}) reached. "
                                "You cannot spawn sub-agents. Research this topic "
                                "directly using web_search and fetch_page instead."
                            }
                        ),
                    }
                )
                continue

            children = registry.spawn_children(agent, queries)
            spawn = PendingSpawn(
                tool_call_id=tc["id"],
                children_ids=[c.id for c in children],
            )
            agent.pending_spawns.append(spawn)
            agent.status = "waiting_for_children"

        elif name == "reference_findings":
            ref_id = args.get("agent_id", "")
            ref_agent = registry.agents.get(ref_id)
            if ref_agent and ref_agent.findings:
                agent.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(
                            {
                                "agent_id": ref_id,
                                "status": ref_agent.status,
                                "findings": ref_agent.findings,
                            }
                        ),
                    }
                )
            else:
                agent.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(
                            {
                                "error": f"Agent {ref_id} not found or has no findings yet.",
                            }
                        ),
                    }
                )

        elif name == "write_report":
            report = args.get("report", "")
            agent.report = report
            agent.findings = report
            agent.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps({"status": "Report saved."}),
                }
            )
            agent.status = "completed"

    return tool_counts


def resolve_waiting_parents(registry: AgentRegistry) -> None:
    """Unblock parents whose children have all completed or failed.

    Each pending spawn resolves independently. A parent may have multiple
    spawn_agents calls pending (e.g., an initial delegation plus a follow-up
    for gaps). Each is resolved as its children complete, and the tool result
    is appended to the parent's messages.

    The parent returns to in_progress only when ALL pending spawns are resolved.
    """
    for agent in list(registry.agents.values()):
        if agent.status != "waiting_for_children":
            continue

        if not agent.pending_spawns:
            agent.status = "in_progress"
            continue

        resolved = []
        for spawn in agent.pending_spawns:
            children = [
                registry.agents[cid]
                for cid in spawn.children_ids
                if cid in registry.agents
            ]
            if not children:
                resolved.append(spawn)
                continue

            all_done = all(c.status in ("completed", "failed") for c in children)
            if not all_done:
                continue

            # Compile children's findings for this spawn
            findings = []
            all_sources = []
            seen_urls = set()
            for child in children:
                # Collect verified URLs from child and all its descendants
                child_urls = registry.collect_verified_urls(child)
                child_source_list = []
                for entry in child_urls:
                    child_source_list.append(entry)
                    if entry["url"] not in seen_urls:
                        seen_urls.add(entry["url"])
                        all_sources.append(entry)

                if child.status == "completed" and child.findings:
                    findings.append(
                        {
                            "agent_id": child.id,
                            "query": child.messages[1]["content"]
                            if len(child.messages) > 1
                            else "",
                            "findings": child.findings,
                            "verified_sources": child_source_list,
                        }
                    )
                else:
                    findings.append(
                        {
                            "agent_id": child.id,
                            "status": child.status,
                            "findings": child.findings
                            or "Agent failed to produce findings.",
                        }
                    )

            result_content = json.dumps(
                {
                    "sub_agent_results": findings,
                    "all_verified_sources": all_sources,
                }
            )
            agent.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": spawn.tool_call_id,
                    "content": result_content,
                }
            )
            resolved.append(spawn)

        # Remove resolved spawns
        for spawn in resolved:
            agent.pending_spawns.remove(spawn)

        # Parent resumes when all spawns are resolved
        if not agent.pending_spawns:
            agent.status = "in_progress"
