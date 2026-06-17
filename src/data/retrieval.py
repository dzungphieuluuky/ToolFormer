"""
retrieval.py
────────────
Two-stage retrieval:

Stage 1 — FunctionRetriever
    Returns the top-k most relevant FUNCTION NAMES for a query.
    Methods: BM25, embedding (sentence-transformers), hybrid.

Stage 2 — ArgumentValueRetriever
    For each parameter in the retrieved functions, finds the most
    relevant ARGUMENT VALUES from a pre-built catalog.
    Example: query "TP.HCM" → location_code: HCM (Thành phố Hồ Chí Minh)

Combined — TelcoRetriever
    Orchestrates both stages and returns a RetrievalResult:
        .function_names     → list[str]
        .argument_values    → dict[param_name → list[ValueMatch]]
"""

from __future__ import annotations

import json
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ValueMatch:
    """A single matched argument value."""

    code: str  # the actual argument value to put in the function call
    label: str  # human-readable name
    group: str  # catalog group (e.g. "Tỉnh/Thành phố", "technology")
    score: float = 0.0  # relevance score (higher = more relevant)
    alt_label: str = ""  # alternative label if present


@dataclass
class RetrievalResult:
    """Complete output of TelcoRetriever.retrieve()."""

    function_names: list[str]  # top-k function names
    argument_values: dict[str, list[ValueMatch]]  # param_name → matches


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Function Retriever (unchanged logic, clean rewrite)
# ──────────────────────────────────────────────────────────────────────────────


class FunctionRetriever:
    """
    Retrieves the top-k most relevant functions for a user query.
    Methods: 'bm25' | 'embedding' | 'hybrid'
    """

    def __init__(
        self,
        function_library: dict,
        method: Literal["bm25", "embedding", "hybrid"] = "hybrid",
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.4,
        emb_weight: float = 0.6,
        index_dir: str | None = None,
    ):
        self.library = function_library
        self.method = method
        self.func_names = list(function_library.keys())
        self.bm25_weight = bm25_weight
        self.emb_weight = emb_weight

        # Rich description: name + description + parameter names
        self.desc_list = [
            self._build_search_text(name, schema)
            for name, schema in function_library.items()
        ]

        self._bm25 = None
        self._encoder = None
        self._embeddings = None

        if method in ("bm25", "hybrid"):
            self._init_bm25()
        if method in ("embedding", "hybrid"):
            self._init_embeddings(encoder_model, index_dir)

    @staticmethod
    def _build_search_text(name: str, schema: dict) -> str:
        """
        Build a rich text representation of a function for indexing.
        Includes name, description, and parameter names/descriptions.
        """
        parts = [name, schema.get("description", "")]
        for pname, pinfo in schema.get("parameters", {}).items():
            parts.append(pname)
            if isinstance(pinfo, dict):
                parts.append(pinfo.get("description", ""))
        tags = schema.get("tags", [])
        parts.extend(tags)
        return " ".join(p for p in parts if p)

    def _init_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [d.lower().split() for d in self.desc_list]
        self._bm25 = BM25Okapi(tokenized)
        print("[FunctionRetriever] BM25 index built.")

    def _init_embeddings(self, model_name: str, index_dir: str | None) -> None:
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(model_name)
        cache = Path(index_dir) / "func_embeddings.npy" if index_dir else None
        if cache and cache.exists():
            self._embeddings = np.load(str(cache))
            print(f"[FunctionRetriever] Loaded embedding cache from {cache}")
        else:
            self._embeddings = self._encoder.encode(
                self.desc_list,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache), self._embeddings)
                print(f"[FunctionRetriever] Embeddings saved → {cache}")

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        scores = self._score(query)
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.func_names[i] for i in top_k]

    def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self._score(query)
        top_k = np.argsort(scores)[-k:][::-1]
        return [(self.func_names[i], float(scores[i])) for i in top_k]

    def _score(self, query: str) -> np.ndarray:
        if self.method == "bm25":
            return self._bm25_scores(query)
        elif self.method == "embedding":
            return self._emb_scores(query)
        else:
            bm25 = self._minmax(self._bm25_scores(query))
            emb = self._minmax(self._emb_scores(query))
            return self.bm25_weight * bm25 + self.emb_weight * emb

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.array(
            self._bm25.get_scores(query.lower().split()),
            dtype=np.float32,
        )

    def _emb_scores(self, query: str) -> np.ndarray:
        q = self._encoder.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        return (self._embeddings @ q).astype(np.float32)

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    def save(self, path: str) -> None:
        state = {
            "func_names": self.func_names,
            "desc_list": self.desc_list,
            "method": self.method,
            "bm25_weight": self.bm25_weight,
            "emb_weight": self.emb_weight,
            "embeddings": self._embeddings,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        print(f"[FunctionRetriever] Saved → {path}")

    @classmethod
    def load(
        cls,
        path: str,
        function_library: dict,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "FunctionRetriever":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.library = function_library
        obj.func_names = state["func_names"]
        obj.desc_list = state["desc_list"]
        obj.method = state["method"]
        obj.bm25_weight = state["bm25_weight"]
        obj.emb_weight = state["emb_weight"]
        obj._embeddings = state["embeddings"]
        obj._bm25 = None
        obj._encoder = None
        if obj.method in ("bm25", "hybrid"):
            obj._init_bm25()
        if obj.method in ("embedding", "hybrid"):
            from sentence_transformers import SentenceTransformer

            obj._encoder = SentenceTransformer(encoder_model)
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Argument Value Retriever
# ──────────────────────────────────────────────────────────────────────────────


class ArgumentValueRetriever:
    """
    For a given query and a set of function parameter names,
    retrieves the most relevant possible argument values from a catalog.

    The catalog maps parameter_name → list of {code, label, group, ...} dicts.

    Matching strategy:
        1. Direct substring match:  query contains code or label verbatim
        2. Normalised fuzzy match:  query normalised (no diacritics) matches
        3. Token match:             any query token matches code/label token

    Scores:
        direct_exact  = 1.0
        direct_fuzzy  = 0.8
        token_match   = 0.5  per matching token (capped at 0.9)
    """

    def __init__(
        self,
        argument_values: dict,  # loaded from argument_values.json
        top_k_values: int = 3,
    ):
        self.catalog = argument_values  # param_name → list[dict]
        self.top_k_values = top_k_values

        # Pre-compute normalised forms for fast matching
        self._norm_cache: dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve_for_function(
        self,
        query: str,
        function_schema: dict,
    ) -> dict[str, list[ValueMatch]]:
        """
        For all parameters in a function schema, retrieve relevant values.

        Parameters
        ──────────
        query           : raw user query string
        function_schema : one function's schema dict from function_library.json

        Returns
        ───────
        {param_name: [ValueMatch, ...]}  — only params with catalog entries
        """
        result: dict[str, list[ValueMatch]] = {}
        params = function_schema.get("parameters", {})

        for param_name in params:
            # Check if this param name has a value catalog
            values_for_param = self._get_catalog(param_name)
            if not values_for_param:
                continue

            # Score all catalog values against the query
            matches = self._score_values(query, values_for_param)
            if matches:
                result[param_name] = matches

        return result

    def retrieve_for_functions(
        self,
        query: str,
        function_names: list[str],
        function_library: dict,
    ) -> dict[str, list[ValueMatch]]:
        """
        Retrieve argument values for all parameters across multiple functions.
        De-duplicates by param_name (union of all retrieved functions).

        Returns
        ───────
        {param_name: [ValueMatch, ...]}
        """
        combined: dict[str, list[ValueMatch]] = {}

        for fn in function_names:
            if fn not in function_library:
                continue
            fn_results = self.retrieve_for_function(query, function_library[fn])
            for param_name, matches in fn_results.items():
                if param_name not in combined:
                    combined[param_name] = matches
                else:
                    # Merge: keep highest-scoring unique codes
                    existing_codes = {m.code for m in combined[param_name]}
                    for m in matches:
                        if m.code not in existing_codes:
                            combined[param_name].append(m)
                    # Re-sort by score
                    combined[param_name].sort(key=lambda x: -x.score)
                    combined[param_name] = combined[param_name][: self.top_k_values]

        return combined

    # ── Catalog lookup ────────────────────────────────────────────────────────

    def _get_catalog(self, param_name: str) -> list[dict]:
        """
        Look up catalog entries for a parameter name.
        Tries exact match first, then suffix/prefix matching
        (e.g. "source_location_code" → "location_code").
        """
        # Exact match
        if param_name in self.catalog:
            return self.catalog[param_name]

        # Suffix match: find the longest catalog key that is a suffix of param_name
        best_key = None
        best_len = 0
        for key in self.catalog:
            if param_name.endswith(key) and len(key) > best_len:
                best_key = key
                best_len = len(key)
        if best_key:
            return self.catalog[best_key]

        # Prefix match: catalog key starts param_name
        for key in self.catalog:
            if param_name.startswith(key):
                return self.catalog[key]

        return []

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_values(
        self,
        query: str,
        catalog: list[dict],
    ) -> list[ValueMatch]:
        """
        Score each catalog entry against the query and return top-k matches.
        Only returns entries with score > 0.
        """
        query_norm = self._normalise(query)
        query_tokens = set(query_norm.split())

        scored: list[ValueMatch] = []

        for entry in catalog:
            code = str(entry.get("code", ""))
            label = str(entry.get("label", ""))
            alt_label = str(entry.get("alt_label", ""))
            group = str(entry.get("group", ""))

            score = self._match_score(
                query,
                query_norm,
                query_tokens,
                code,
                label,
                alt_label,
            )

            if score > 0.0:
                scored.append(
                    ValueMatch(
                        code=code,
                        label=label,
                        group=group,
                        score=score,
                        alt_label=alt_label,
                    )
                )

        # Sort by score descending, return top-k
        scored.sort(key=lambda x: -x.score)
        return scored[: self.top_k_values]

    def _match_score(
        self,
        query: str,
        query_norm: str,
        query_tokens: set[str],
        code: str,
        label: str,
        alt_label: str,
    ) -> float:
        """
        Compute relevance score for one catalog entry.

        Scoring rules (highest score wins):
          1.0  exact code match (case-insensitive)
          1.0  exact label match (case-insensitive, normalised)
          0.8  code appears as substring in query
          0.8  label appears as substring in query (normalised)
          0.8  alt_label appears as substring in query (normalised)
          0.5  per overlapping token between query tokens and label tokens
               (capped at 0.9)
        """
        q_lower = query.lower()
        code_lower = code.lower()

        # Rule 1: exact code in query
        if code_lower in q_lower:
            # Weight: longer codes are more specific matches
            specificity = min(1.0, len(code) / 5.0)
            return 0.8 + 0.2 * specificity

        # Rule 2: exact label in query (normalised)
        label_norm = self._normalise(label)
        alt_label_norm = self._normalise(alt_label) if alt_label else ""

        if label_norm in query_norm:
            return 1.0
        if alt_label_norm and alt_label_norm in query_norm:
            return 0.9

        # Rule 3: token overlap (normalised)
        label_tokens = set(label_norm.split())
        alt_label_tokens = set(alt_label_norm.split()) if alt_label_norm else set()
        all_label_tokens = label_tokens | alt_label_tokens

        # Remove very short/common tokens
        meaningful_label_tokens = {t for t in all_label_tokens if len(t) >= 2}
        meaningful_query_tokens = {t for t in query_tokens if len(t) >= 2}

        if not meaningful_label_tokens:
            return 0.0

        overlap = meaningful_query_tokens & meaningful_label_tokens
        if overlap:
            token_score = min(0.7, 0.35 * len(overlap))
            return token_score

        return 0.0

    def _normalise(self, text: str) -> str:
        """
        Normalise text for matching:
        - Lowercase
        - Remove Vietnamese diacritics
        - Collapse whitespace
        """
        if text in self._norm_cache:
            return self._norm_cache[text]

        # Remove diacritics via Unicode normalisation
        nfkd = unicodedata.normalize("NFKD", text.lower())
        ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))

        # Extra Vietnamese normalisation
        replacements = {
            "đ": "d",
            "ð": "d",
        }
        for src, dst in replacements.items():
            ascii_text = ascii_text.replace(src, dst)

        result = re.sub(r"\s+", " ", ascii_text).strip()
        self._norm_cache[text] = result
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Combined TelcoRetriever
# ──────────────────────────────────────────────────────────────────────────────


class TelcoRetriever:
    """
    Orchestrates FunctionRetriever + ArgumentValueRetriever.

    Usage
    ─────
    retriever = TelcoRetriever.build(function_library, argument_values)
    result    = retriever.retrieve("Xem KPI tại TP.HCM", k=5)

    result.function_names   → ["SPEEDTEST_PROVINCE", ...]
    result.argument_values  → {"location_code": [ValueMatch(code="HCM", ...)], ...}
    """

    def __init__(
        self,
        function_retriever: FunctionRetriever,
        value_retriever: ArgumentValueRetriever,
    ):
        self.func_retriever = function_retriever
        self.value_retriever = value_retriever

    @classmethod
    def build(
        cls,
        function_library: dict,
        argument_values: dict,
        method: str = "hybrid",
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k_values: int = 3,
        index_dir: str | None = None,
    ) -> "TelcoRetriever":
        """Build a TelcoRetriever from raw library and value catalog dicts."""
        func_ret = FunctionRetriever(
            function_library=function_library,
            method=method,
            encoder_model=encoder_model,
            index_dir=index_dir,
        )
        val_ret = ArgumentValueRetriever(
            argument_values=argument_values,
            top_k_values=top_k_values,
        )
        return cls(func_ret, val_ret)

    @classmethod
    def load(
        cls,
        retriever_path: str,
        function_library: dict,
        argument_values: dict,
        top_k_values: int = 3,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "TelcoRetriever":
        """Load a saved FunctionRetriever + build a fresh ArgumentValueRetriever."""
        func_ret = FunctionRetriever.load(
            retriever_path, function_library, encoder_model
        )
        val_ret = ArgumentValueRetriever(
            argument_values=argument_values,
            top_k_values=top_k_values,
        )
        return cls(func_ret, val_ret)

    def retrieve(
        self,
        query: str,
        function_library: dict,
        k: int = 5,
        precomputed_func_names: list[str] | None = None,
    ) -> RetrievalResult:
        """
        Full two-stage retrieval.

        Parameters
        ──────────
        query                  : user query
        function_library       : full schema dict
        k                      : number of functions to retrieve
        precomputed_func_names : if provided, skip function retrieval
                                 (used when dataset already has retrieved_functions)

        Returns
        ───────
        RetrievalResult with function_names and argument_values
        """
        # Stage 1: function retrieval
        if precomputed_func_names is not None:
            func_names = precomputed_func_names
        else:
            func_names = self.func_retriever.retrieve(query, k=k)

        # Stage 2: argument value retrieval
        arg_values = self.value_retriever.retrieve_for_functions(
            query=query,
            function_names=func_names,
            function_library=function_library,
        )

        return RetrievalResult(
            function_names=func_names,
            argument_values=arg_values,
        )
