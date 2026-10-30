"""Tool definitions and a registry supporting tool calling."""
import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]

    def __call__(self, **kwargs):
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return list(self._tools)

    def describe(self) -> List[dict]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def call(self, name: str, arguments: dict) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](**arguments)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculator(expression: str) -> float:
    """Safely evaluate an arithmetic expression."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("Unsupported expression")
    return ev(ast.parse(expression, mode="eval").body)


def default_registry(search_fn: Callable[[str], List[str]]) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("calculator", "Evaluate an arithmetic expression", calculator))
    reg.register(Tool("search", "Retrieve documents relevant to a query",
                      lambda query: search_fn(query)))
    return reg
