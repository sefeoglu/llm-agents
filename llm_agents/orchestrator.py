"""Orchestration: routes between LLM, tools and the reranker."""
from typing import List, Optional

from .llm import LLM
from .reranker import Reranker
from .tools import ToolRegistry, default_registry


class Agent:
    """Loop: decide one action, observe its result, then re-plan."""

    def __init__(self, llm: LLM, tools: ToolRegistry, reranker: Reranker,
                 max_steps: int = 5, top_k: int = 3):
        self.llm, self.tools, self.reranker = llm, tools, reranker
        self.max_steps, self.top_k = max_steps, top_k

    def run(self, question: str) -> dict:
        obs: List[dict] = []
        tool_calls_used = 0
        while True:
            d = self.llm.decide(question, obs, self.tools.describe())
            if "answer" in d:
                return {"answer": d["answer"], "trace": obs}
            calls = d.get("tool_calls")
            if calls is None:
                calls = [{
                    "tool": d.get("tool"),
                    "arguments": d.get("arguments", {}),
                    "tool_call_id": d.get("tool_call_id"),
                }]
            if not isinstance(calls, list) or not calls:
                raise ValueError("LLM returned an invalid or empty tool-call list")
            if len(calls) != 1:
                raise ValueError(
                    "LLM must choose one tool per decision so it can re-plan "
                    "after observing each result"
                )
            if tool_calls_used >= self.max_steps:
                return {"answer": "Max steps reached.", "trace": obs}
            call = calls[0]
            name = call.get("tool")
            if name not in self.tools.names():
                raise ValueError(f"LLM selected an unavailable tool: {name}")
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError(f"LLM returned invalid arguments for tool {name}")
            result = self.tools.call(name, arguments)
            if name == "search":
                result = self.reranker.rerank(
                    arguments["query"], result, self.top_k
                )
            observation = {
                "tool": name, "arguments": arguments, "result": result,
            }
            if call.get("tool_call_id"):
                observation["tool_call_id"] = call["tool_call_id"]
            obs.append(observation)
            tool_calls_used += 1


class Orchestrator(Agent):
    """Adaptively choose tools over a corpus, observing each result before replanning."""

    def __init__(self, llm: LLM, corpus: List[str], reranker: Optional[Reranker] = None,
                 max_steps: int = 5, top_k: int = 3):
        self.corpus = corpus
        super().__init__(
            llm,
            default_registry(self._retrieve),
            reranker or Reranker(),
            max_steps=max_steps,
            top_k=top_k,
        )

    def _retrieve(self, query: str) -> List[str]:
        # First stage: keep any doc sharing a token with the query; reranker orders them.
        from .reranker import tokenize
        q = set(tokenize(query))
        return [d for d in self.corpus if q & set(tokenize(d))]
