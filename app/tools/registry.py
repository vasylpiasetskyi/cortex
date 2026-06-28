import json
from collections.abc import Callable

from app.tools.calculator import calculate
from app.tools.weather import get_weather

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Warsaw"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use for any arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '2 + 2 * 5' or '(10 + 5) / 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable] = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def execute_tool(name: str, arguments_json: str) -> str:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        args = json.loads(arguments_json)
        return fn(**args)
    except Exception as exc:
        return f"Tool execution error: {exc}"
