"""Tool definitions and execution for recursive research agents.

Tools are defined in OpenAI function-calling format and executed locally
between batch rounds. The model decides which tools to call and when.

Three tools are "deferred" — spawn_agents, write_report, and
reference_findings — meaning they are handled by the orchestrator
rather than executed immediately.
"""

import json

from .scrape import fetch_urls
from .search import search

# Sentinel value for tools handled by the orchestrator, not here
DEFERRED = "__DEFERRED__"

_REFERENCE_FINDINGS = {
    "type": "function",
    "function": {
        "name": "reference_findings",
        "description": (
            "Reference the findings of another agent that has already "
            "researched a similar or related topic. Use this instead of "
            "re-searching a topic that another agent has already covered. "
            "Check the active_agents list in your context to see what "
            "topics are already being researched."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the agent whose findings to reference",
                },
            },
            "required": ["agent_id"],
        },
    },
}

_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the web for a specific angle or follow-up query. "
            "Your topic was already searched when you were created and "
            "results are in your context — use this only to explore "
            "DIFFERENT angles or follow-up questions, not to repeat "
            "your initial search. Prefer spawning sub-agents over "
            "calling search multiple times — each sub-agent gets its "
            "own automatic search and works in parallel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

_READ_PAGES = {
    "type": "function",
    "function": {
        "name": "read_pages",
        "description": (
            "Read one or more web pages in parallel. Returns each page's "
            "content as markdown text (truncated to 15000 chars each). "
            "Pass ALL the URLs you want to read in a single call — they "
            "are fetched simultaneously. Use this to read promising URLs "
            "from your search results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to fetch and read",
                },
            },
            "required": ["urls"],
        },
    },
}

_SPAWN_AGENTS = {
    "type": "function",
    "function": {
        "name": "spawn_agents",
        "description": (
            "Spawn parallel sub-agents to research different topics "
            "independently. Each sub-agent automatically gets web search "
            "results for its topic and can then read pages, search for "
            "new angles, or spawn its own sub-agents. Returns their "
            "combined findings when all complete. Prefer this over "
            "calling search multiple times — sub-agents work in parallel "
            "and are more efficient."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of research topics/queries, one per sub-agent"
                    ),
                },
            },
            "required": ["queries"],
        },
    },
}

_WRITE_REPORT = {
    "type": "function",
    "function": {
        "name": "write_report",
        "description": (
            "Write the final research report. Call this when you have "
            "gathered all findings from your sub-agents and any additional "
            "research, and are ready to produce the final output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "report": {
                    "type": "string",
                    "description": "The complete research report in markdown",
                },
            },
            "required": ["report"],
        },
    },
}

# Root agent gets all tools
ROOT_TOOLS = [
    _SEARCH,
    _READ_PAGES,
    _SPAWN_AGENTS,
    _REFERENCE_FINDINGS,
    _WRITE_REPORT,
]

# Sub-agents get all except write_report
SUB_AGENT_TOOLS = [_SEARCH, _READ_PAGES, _SPAWN_AGENTS, _REFERENCE_FINDINGS]


def get_tools_for_agent(is_root: bool) -> list[dict]:
    """Return the appropriate tool definitions for an agent."""
    return ROOT_TOOLS if is_root else SUB_AGENT_TOOLS


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a JSON string.

    Deferred tools (spawn_agents, write_report, reference_findings) return
    DEFERRED — they are handled by the orchestrator, not here.

    Args:
        name: Tool function name
        arguments: JSON string of arguments

    Returns:
        JSON string with the tool result, or DEFERRED sentinel
    """
    if name in ("spawn_agents", "write_report", "reference_findings"):
        return DEFERRED

    args = json.loads(arguments)

    if name == "search":
        result = search(args["query"], max_results=args.get("max_results", 5))
        return json.dumps(result)

    if name == "read_pages":
        urls = args.get("urls", [])
        if not urls:
            return json.dumps({"error": "No URLs provided"})
        fetched = fetch_urls(urls)
        pages = []
        for url in urls:
            content = fetched.get(url)
            if content:
                pages.append({"url": url, "content": content[:15000]})
            else:
                pages.append({"url": url, "error": f"Failed to fetch {url}"})
        return json.dumps({"pages": pages})

    return json.dumps({"error": f"Unknown tool: {name}"})
