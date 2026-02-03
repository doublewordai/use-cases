"""Tool definitions and execution for research agents.

Tools are defined in OpenAI function-calling format and executed locally
between batch rounds. The model decides which tools to call and when.
"""

import json

from .scrape import fetch_url
from .search import search

TOOL_DEFINITIONS = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch and read a web page. Returns the page content as "
                "markdown text (truncated to 15000 chars). Use this after "
                "web_search to read promising results in detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to fetch",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a JSON string.

    Args:
        name: Tool function name (web_search or fetch_page)
        arguments: JSON string of arguments

    Returns:
        JSON string with the tool result
    """
    args = json.loads(arguments)

    if name == "web_search":
        result = search(args["query"], max_results=args.get("max_results", 5))
        return json.dumps(result)

    if name == "fetch_page":
        content = fetch_url(args["url"])
        if content is None:
            return json.dumps({"error": f"Failed to fetch {args['url']}"})
        # Truncate to keep context manageable
        return json.dumps({"url": args["url"], "content": content[:15000]})

    return json.dumps({"error": f"Unknown tool: {name}"})
