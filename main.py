from llm_agents import Orchestrator, RuleBasedLLM

CORPUS = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is located in Paris.",
]

if __name__ == "__main__":
    o = Orchestrator(RuleBasedLLM(), CORPUS)
    for q in ["What is the capital of France?", "Compute 12 * (3 + 4)"]:
        print(q, "->", o.run(q)["answer"])
