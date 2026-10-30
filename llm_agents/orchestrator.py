"""Orchestration: routes between LLM, tools and the reranker."""
from typing import List, Optional

from .llm import LLM
from .reranker import Reranker
from .tools import ToolRegistry, default_registry


class Agent:
    """Loop: LLM decides -> tool call -> observation -> ... -> answer."""

    def __init__(self, llm: LLM, tools: ToolRegistry, reranker: Reranker,
                 max_steps: int = 5, top_k: int = 3):
        self.llm, self.tools, self.reranker = llm, tools, reranker
        self.max_steps, self.top_k = max_steps, top_k

    def run(self, question: str) -> dict:
        obs: List[dict] = []
        for _ in range(self.max_steps):
            d = self.llm.decide(question, obs)
            if "answer" in d:
                return {"answer": d["answer"], "trace": obs}
            result = self.tools.call(d["tool"], d["arguments"])
            if d["tool"] == "search":
                result = self.reranker.rerank(d["arguments"]["query"], result, self.top_k)
            obs.append({"tool": d["tool"], "arguments": d["arguments"], "result": result})
        return {"answer": "Max steps reached.", "trace": obs}


class Orchestrator:
    """Builds an agent over a document corpus (candidate retrieval + reranking)."""

    def __init__(self, llm: LLM, corpus: List[str], reranker: Optional[Reranker] = None):
        self.corpus = corpus
        self.agent = Agent(llm, default_registry(self._retrieve), reranker or Reranker())

    def _retrieve(self, query: str) -> List[str]:
        # First stage: keep any doc sharing a token with the query; reranker orders them.
        from .reranker import tokenize
        q = set(tokenize(query))
        return [d for d in self.corpus if q & set(tokenize(d))]

    def run(self, question: str) -> dict:
        return self.agent.run(question)
