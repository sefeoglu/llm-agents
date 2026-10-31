from llm_agents import LLM, Orchestrator, Reranker
from llm_agents.orchestrator import Agent
from llm_agents.tools import calculator

CORPUS = ["Paris is the capital of France.", "Berlin is the capital of Germany.",
          "The Eiffel Tower is in Paris."]


class ScriptedLLM(LLM):
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def decide(self, question, observations, tools=None):
        return next(self.decisions)


def test_calculator():
    assert calculator("2+3*4") == 14


def test_rerank_orders_relevant_first():
    r = Reranker().rerank("capital of France", CORPUS, top_k=2)
    assert r[0][0] == CORPUS[0]


def test_tool_call_math():
    llm = ScriptedLLM([
        {"tool": "calculator", "arguments": {"expression": "12 * (3 + 4)"}},
        {"answer": "84"},
    ])
    assert Orchestrator(llm, CORPUS).run("Compute 12 * (3 + 4)")["answer"] == "84"


def test_search_rerank():
    llm = ScriptedLLM([
        {"tool": "search", "arguments": {"query": "capital of France"}},
        {"answer": CORPUS[0]},
    ])
    out = Orchestrator(llm, CORPUS).run("What is the capital of France?")
    assert out["answer"] == CORPUS[0]
    assert out["trace"][0]["tool"] == "search"


def test_agent_replans_after_each_tool_observation():
    class AdaptiveLLM(LLM):
        def __init__(self):
            self.observation_counts = []

        def decide(self, question, observations, tools=None):
            self.observation_counts.append(len(observations))
            if not observations:
                return {
                    "tool": "calculator",
                    "arguments": {"expression": "6 * 7"},
                }
            if len(observations) == 1:
                assert observations[0]["result"] == 42
                return {
                    "tool": "calculator",
                    "arguments": {"expression": "42 * 2"},
                }
            assert observations[1]["result"] == 84
            return {"answer": "The first result was doubled to 84."}

    llm = AdaptiveLLM()
    orchestrator = Orchestrator(llm, CORPUS, max_steps=2)
    result = orchestrator.run("Compute 6 * 7, then double the result")

    assert isinstance(orchestrator, Agent)
    assert llm.observation_counts == [0, 1, 2]
    assert [item["result"] for item in result["trace"]] == [42, 84]
    assert result["answer"] == "The first result was doubled to 84."


def test_search_returns_no_unrelated_documents_when_no_terms_match():
    llm = ScriptedLLM([
        {"tool": "search", "arguments": {"query": "quantum chromodynamics"}},
        {"answer": "I could not find supporting evidence in the indexed corpus."},
    ])
    result = Orchestrator(llm, CORPUS).run("Find quantum chromodynamics papers")

    assert result["trace"][0]["result"] == []
    assert "could not find supporting evidence" in result["answer"]
