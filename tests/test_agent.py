from llm_agents import Orchestrator, RuleBasedLLM, Reranker
from llm_agents.tools import calculator

CORPUS = ["Paris is the capital of France.", "Berlin is the capital of Germany.",
          "The Eiffel Tower is in Paris."]


def test_calculator():
    assert calculator("2+3*4") == 14


def test_rerank_orders_relevant_first():
    r = Reranker().rerank("capital of France", CORPUS, top_k=2)
    assert r[0][0] == CORPUS[0]


def test_tool_call_math():
    assert Orchestrator(RuleBasedLLM(), CORPUS).run("Compute 12 * (3 + 4)")["answer"] == "84"


def test_search_rerank():
    out = Orchestrator(RuleBasedLLM(), CORPUS).run("What is the capital of France?")
    assert out["answer"] == CORPUS[0]
    assert out["trace"][0]["tool"] == "search"
