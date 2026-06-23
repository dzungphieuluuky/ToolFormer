from __future__ import annotations

import re
import json
import calendar
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from rank_bm25 import BM25Okapi

from src.utils.vietnamese_normalizer import normalize_vietnamese, expand_synonyms, tokenize_meaningful, build_ngrams
from src.utils.value_catalog import ValueCatalog, ValueMatch, CatalogEntry

_DATE_PATTERNS = [
    (
        r"(?:thang|tháng)\s*(\d{1,2})\s*[/\-\.]\s*(\d{4})",
        lambda m: (
            f"{m.group(2)}-{int(m.group(1)):02d}-01",
            _last_day(int(m.group(2)), int(m.group(1))),
        ),
    ),
    (
        r"(?:thang|tháng)\s*(\d{1,2})\s*(?:nam|năm)\s*(\d{4})",
        lambda m: (
            f"{m.group(2)}-{int(m.group(1)):02d}-01",
            _last_day(int(m.group(2)), int(m.group(1))),
        ),
    ),
    (
        r"(?:toan\s*)?(?:nam|năm)\s*(\d{4})",
        lambda m: (f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"),
    ),
    (
        r"(?:q|quy|quý)\s*(\d)\s*[/\-]?\s*(\d{4})",
        lambda m: _quarter_range(int(m.group(1)), int(m.group(2))),
    ),
    (
        r"(\d{1,2})\s*/\s*(\d{4})",
        lambda m: (
            f"{m.group(2)}-{int(m.group(1)):02d}-01",
            _last_day(int(m.group(2)), int(m.group(1))),
        ),
    ),
    (
        r"(\d{4}-\d{2}-\d{2})\s*(?:den|đến|to|\-)\s*(\d{4}-\d{2}-\d{2})",
        lambda m: (m.group(1), m.group(2)),
    ),
]


def _last_day(year: int, month: int) -> str:
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last:02d}"


def _quarter_range(q: int, year: int) -> tuple[str, str]:
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return (f"{year}-{starts[q]}", f"{year}-{ends[q]}")


def extract_dates(query: str) -> dict[str, str]:
    query_lower = query.lower()
    for pattern, extractor in _DATE_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            from_date, to_date = extractor(match)
            return {"from_date": from_date, "to_date": to_date}
    return {}


_DATA_LEVEL_PATTERNS = [
    (r"(?:theo|tong\s*hop)\s*(?:ngay|ngày|daily)", "day"),
    (r"(?:theo|tong\s*hop)\s*(?:tuan|tuần|weekly)", "week"),
    (r"(?:theo|tong\s*hop)\s*(?:thang|tháng|monthly)", "month"),
    (r"(?:theo|tong\s*hop)\s*(?:nam|năm|yearly)", "year"),
    (r"(?:hang|hàng)\s*(?:ngay|ngày)", "day"),
    (r"(?:hang|hàng)\s*(?:tuan|tuần)", "week"),
    (r"(?:hang|hàng)\s*(?:thang|tháng)", "month"),
    (r"(?:thang|tháng)\s*\d{1,2}", "month"),
]


def extract_data_level(query: str) -> str | None:
    query_norm = normalize_vietnamese(query)
    for pattern, level in _DATA_LEVEL_PATTERNS:
        if re.search(pattern, query_norm):
            return level
    return None


class FunctionRetriever:
    def __init__(
        self,
        function_library: dict,
        method: Literal["bm25", "hybrid"] = "bm25",
        encoder_model: str | None = None,
        bm25_weight: float = 0.5,
        emb_weight: float = 0.5,
    ):
        self.library = function_library
        self.method = method
        self.func_names = list(function_library.keys())
        self.bm25_weight = bm25_weight
        self.emb_weight = emb_weight

        self._search_texts = []
        self._search_tokens = []
        for name, schema in function_library.items():
            text = self._build_search_text(name, schema)
            norm = normalize_vietnamese(text)
            expanded = expand_synonyms(norm)
            self._search_texts.append(expanded)
            self._search_tokens.append(expanded.split())

        self._bm25 = BM25Okapi(self._search_tokens)

        self._encoder = None
        self._embeddings = None
        if method == "hybrid" and encoder_model:
            self._init_embeddings(encoder_model)

    @staticmethod
    def _build_search_text(name: str, schema: dict) -> str:
        parts = [name]
        parts.append(schema.get("description", ""))
        for pname, pinfo in schema.get("parameters", {}).items():
            parts.append(pname)
            if isinstance(pinfo, dict):
                parts.append(pinfo.get("description", ""))
        parts.extend(schema.get("tags", []))
        parts.extend(schema.get("examples", []))
        return " ".join(p for p in parts if p)

    def _init_embeddings(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(model_name)
        self._embeddings = self._encoder.encode(
            self._search_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        scores = self._score(query)
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.func_names[i] for i in top_k]

    def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self._score(query)
        top_k = np.argsort(scores)[-k:][::-1]
        return [(self.func_names[i], float(scores[i])) for i in top_k]

    def _score(self, query: str) -> np.ndarray:
        query_norm = expand_synonyms(normalize_vietnamese(query))
        if self.method == "bm25" or self._encoder is None:
            return self._bm25_scores(query_norm)
        bm25 = self._minmax(self._bm25_scores(query_norm))
        emb = self._minmax(self._emb_scores(query))
        return self.bm25_weight * bm25 + self.emb_weight * emb

    def _bm25_scores(self, query_normalized: str) -> np.ndarray:
        tokens = query_normalized.split()
        return np.array(self._bm25.get_scores(tokens), dtype=np.float32)

    def _emb_scores(self, query: str) -> np.ndarray:
        q = self._encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        return (self._embeddings @ q).astype(np.float32)

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)


class ArgumentValueRetriever:
    def __init__(
        self,
        catalogs: dict[str, ValueCatalog],
        top_k_values: int = 5,
        include_all_threshold: int = 12,
    ):
        self.catalogs = catalogs
        self.top_k = top_k_values
        self.include_all_threshold = include_all_threshold

        self._param_to_catalog: dict[str, str] = {}
        for key in catalogs:
            self._param_to_catalog[key] = key

    def add_param_mapping(self, param_name: str, catalog_key: str) -> None:
        self._param_to_catalog[param_name] = catalog_key

    def retrieve_for_function(
        self,
        query: str,
        function_schema: dict,
    ) -> dict[str, list[ValueMatch]]:
        result: dict[str, list[ValueMatch]] = {}
        params = function_schema.get("parameters", {})

        query_norm = expand_synonyms(normalize_vietnamese(query))
        query_tokens = tokenize_meaningful(query_norm)

        for param_name in params:
            if param_name in ("from_date", "to_date"):
                dates = extract_dates(query)
                if param_name in dates:
                    result[param_name] = [
                        ValueMatch(code=dates[param_name], label=dates[param_name],
                                   group="extracted_date", score=1.0)
                    ]
                continue

            if param_name == "data_level":
                level = extract_data_level(query)
                if level:
                    result[param_name] = [
                        ValueMatch(code=level, label=level, group="extracted_level", score=1.0)
                    ]
                catalog = self._get_catalog(param_name)
                if catalog:
                    if catalog.size <= self.include_all_threshold:
                        all_vals = catalog.get_all()
                        if param_name in result:
                            existing_codes = {m.code for m in result[param_name]}
                            for v in all_vals:
                                if v.code not in existing_codes:
                                    result[param_name].append(v)
                        else:
                            result[param_name] = all_vals
                continue

            catalog = self._get_catalog(param_name)
            if catalog is None:
                constraints = function_schema.get("constraints", {})
                param_con = constraints.get(param_name, {})
                if "enum" in param_con:
                    result[param_name] = [
                        ValueMatch(code=str(v), label=str(v), group="enum", score=1.0)
                        for v in param_con["enum"]
                    ]
                continue

            if catalog.size <= self.include_all_threshold:
                result[param_name] = catalog.get_all()
            else:
                matches = catalog.match(query_norm, query_tokens, top_k=self.top_k)
                if matches:
                    result[param_name] = matches

        return result

    def retrieve_for_functions(
        self,
        query: str,
        function_names: list[str],
        function_library: dict,
    ) -> dict[str, list[ValueMatch]]:
        combined: dict[str, list[ValueMatch]] = {}
        for fn in function_names:
            if fn not in function_library:
                continue
            fn_results = self.retrieve_for_function(query, function_library[fn])
            for param_name, matches in fn_results.items():
                if param_name not in combined:
                    combined[param_name] = matches
                else:
                    existing_codes = {m.code for m in combined[param_name]}
                    for m in matches:
                        if m.code not in existing_codes:
                            combined[param_name].append(m)
                    combined[param_name].sort(key=lambda x: -x.score)
                    combined[param_name] = combined[param_name][:self.top_k]
        return combined

    def _get_catalog(self, param_name: str) -> ValueCatalog | None:
        catalog_key = self._param_to_catalog.get(param_name)
        if catalog_key and catalog_key in self.catalogs:
            return self.catalogs[catalog_key]
        if param_name in self.catalogs:
            return self.catalogs[param_name]
        best_key = None
        best_len = 0
        for key in self.catalogs:
            if param_name.endswith(key) and len(key) > best_len:
                best_key = key
                best_len = len(key)
        if best_key:
            return self.catalogs[best_key]
        for key in self.catalogs:
            if param_name.startswith(key):
                return self.catalogs[key]
        return None


@dataclass
class RetrievalResult:
    function_names: list[str]
    argument_values: dict[str, list[ValueMatch]]
    extracted_dates: dict[str, str]
    extracted_data_level: str | None


class TelcoRetriever:
    def __init__(
        self,
        func_retriever: FunctionRetriever,
        value_retriever: ArgumentValueRetriever,
    ):
        self.func_retriever = func_retriever
        self.value_retriever = value_retriever

    @classmethod
    def build(
        cls,
        function_library: dict,
        catalogs: dict[str, ValueCatalog],
        method: Literal["bm25", "hybrid"] = "bm25",
        encoder_model: str | None = None,
        top_k_values: int = 5,
        include_all_threshold: int = 12,
    ) -> "TelcoRetriever":
        func_ret = FunctionRetriever(function_library, method=method, encoder_model=encoder_model)
        val_ret = ArgumentValueRetriever(catalogs, top_k_values=top_k_values,
                                         include_all_threshold=include_all_threshold)
        return cls(func_ret, val_ret)

    def retrieve(
        self,
        query: str,
        function_library: dict,
        k: int = 5,
        precomputed_func_names: list[str] | None = None,
    ) -> RetrievalResult:
        if precomputed_func_names is not None:
            func_names = precomputed_func_names
        else:
            func_names = self.func_retriever.retrieve(query, k=k)

        arg_values = self.value_retriever.retrieve_for_functions(
            query, func_names, function_library
        )

        dates = extract_dates(query)
        data_level = extract_data_level(query)

        return RetrievalResult(
            function_names=func_names,
            argument_values=arg_values,
            extracted_dates=dates,
            extracted_data_level=data_level,
        )
