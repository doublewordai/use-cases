"""Tool definitions and execution for recursive research agents.

Tools are defined in OpenAI function-calling format and executed locally
between batch rounds. The model decides which tools to call and when.

Two tools are "deferred" — spawn_agents and write_report — meaning they
are handled by the orchestrator rather than executed immediately.
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

_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for information on a topic. Returns a list of "
            "results with titles, URLs, and snippets."
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

_FETCH_PAGES = {
    "type": "function",
    "function": {
        "name": "fetch_pages",
        "description": (
            "Fetch and read one or more web pages in parallel. Returns "
            "each page's content as markdown text (truncated to 15000 "
            "chars each). Call this with all the URLs you want to read "
            "at once — they are fetched simultaneously."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to fetch",
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
            "independently. Each query becomes a separate research agent "
            "that can search the web, read pages, and even spawn its own "
            "sub-agents. Returns their combined findings when all complete."
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
    _WEB_SEARCH,
    _FETCH_PAGES,
    _SPAWN_AGENTS,
    _REFERENCE_FINDINGS,
    _WRITE_REPORT,
]

# Sub-agents get all except write_report
SUB_AGENT_TOOLS = [_WEB_SEARCH, _FETCH_PAGES, _SPAWN_AGENTS, _REFERENCE_FINDINGS]


def get_tools_for_agent(is_root: bool) -> list[dict]:
    """Return the appropriate tool definitions for an agent."""
    return ROOT_TOOLS if is_root else SUB_AGENT_TOOLS


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a JSON string.

    Deferred tools (spawn_agents, write_report) return DEFERRED — they are
    handled by the orchestrator in agent.py, not here.

    Args:
        name: Tool function name
        arguments: JSON string of arguments

    Returns:
        JSON string with the tool result, or DEFERRED sentinel
    """
    if name in ("spawn_agents", "write_report", "reference_findings"):
        return DEFERRED

    args = json.loads(arguments)

    if name == "web_search":
        result = search(args["query"], max_results=args.get("max_results", 5))
        return json.dumps(result)

    if name == "fetch_pages":
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
