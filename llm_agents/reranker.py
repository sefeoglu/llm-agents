"""Simple BM25 reranker for candidate documents."""
import math
import re
from collections import Counter
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class Reranker:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def rerank(self, query: str, docs: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        toks = [tokenize(d) for d in docs]
        if not toks:
            return []
        n = len(toks)
        avg = sum(len(t) for t in toks) / n or 1.0
        df = Counter(w for t in toks for w in set(t))
        scored = []
        for doc, t in zip(docs, toks):
            tf = Counter(t)
            s = 0.0
            for w in tokenize(query):
                if w in tf:
                    idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                    s += idf * tf[w] * (self.k1 + 1) / (
                        tf[w] + self.k1 * (1 - self.b + self.b * len(t) / avg))
            scored.append((doc, s))
        return sorted(scored, key=lambda x: -x[1])[:top_k]
