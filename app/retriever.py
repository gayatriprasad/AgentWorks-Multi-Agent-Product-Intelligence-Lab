# app/retriever.py
from typing import List, Tuple
from app.store import get_all_products
import re
from collections import Counter
from math import log, sqrt

def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())

class LocalTfidfVectorizer:
    def __init__(self, max_features: int = 40_000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocab = {}
        self.idf = {}
        self.n_docs = 0

    def _ngrams(self, tokens):
        lo, hi = self.ngram_range
        grams = []
        for n in range(lo, hi + 1):
            if n == 1:
                grams.extend(tokens)
            else:
                grams.extend([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
        return grams

    def fit(self, texts: List[str]):
        self.n_docs = len(texts)
        df = Counter()
        for t in texts:
            toks = self._ngrams(_tokenize(t))
            df.update(set(toks))
        most_common = df.most_common(self.max_features)
        self.vocab = {term: i for i, (term, _) in enumerate(most_common)}
        self.idf = {term: log((1 + self.n_docs) / (1 + df[term])) + 1.0 for term in self.vocab}
        return self

    def _tfidf_vec(self, text: str):
        toks = self._ngrams(_tokenize(text))
        tf = Counter([t for t in toks if t in self.vocab])
        return {t: tf[t] * self.idf[t] for t in tf}

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return [self._tfidf_vec(t) for t in texts]

    def transform(self, texts: List[str]):
        return [self._tfidf_vec(t) for t in texts]

def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    na = sqrt(sum(w * w for w in a.values()))
    nb = sqrt(sum(w * w for w in b.values()))
    return (dot / (na * nb)) if na and nb else 0.0

# Try sklearn first; fall back if unavailable
try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SKTfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False
    SKTfidfVectorizer = None

class SearchIndex:
    def __init__(self):
        self.product_ids: List[str] = []
        self.using_sklearn = _HAVE_SKLEARN
        self.vectorizer = (SKTfidfVectorizer(max_features=40_000, ngram_range=(1, 2))
                           if self.using_sklearn else LocalTfidfVectorizer(max_features=40_000, ngram_range=(1, 2)))
        self.matrix = None     # sklearn mode
        self.doc_vecs = None   # fallback mode
        self._fit()

    def _fit(self):
        prods = get_all_products()
        corpus = [(p.product_id, f"{p.title} {p.description or ''} {p.category or ''}") for p in prods]
        self.product_ids = [pid for pid, _ in corpus]
        texts = [txt for _, txt in corpus]
        if not texts:
            return
        if self.using_sklearn:
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.doc_vecs = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.product_ids:
            return []
        if self.using_sklearn:
            qv = self.vectorizer.transform([query])
            sims = linear_kernel(qv, self.matrix).ravel()
            idxs = sims.argsort()[::-1][:top_k]
            return [(self.product_ids[i], float(sims[i])) for i in idxs]
        else:
            qv = self.vectorizer.transform([query])[0]
            sims = [(_cosine(qv, dv), i) for i, dv in enumerate(self.doc_vecs)]
            sims.sort(key=lambda x: x[0], reverse=True)
            top = sims[:top_k]
            return [(self.product_ids[i], float(s)) for (s, i) in top]
