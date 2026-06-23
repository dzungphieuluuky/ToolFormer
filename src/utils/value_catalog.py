from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.vietnamese_normalizer import normalize_vietnamese, expand_synonyms, tokenize_meaningful, build_ngrams


@dataclass
class CatalogEntry:
    code: str
    label: str
    group: str = ""
    alt_label: str = ""
    _norm_label: str = field(init=False, repr=False, default="")
    _norm_alt: str = field(init=False, repr=False, default="")
    _norm_code: str = field(init=False, repr=False, default="")
    _label_tokens: frozenset = field(init=False, repr=False, default_factory=frozenset)
    _aliases: list[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        self._norm_label = normalize_vietnamese(self.label)
        self._norm_alt = normalize_vietnamese(self.alt_label) if self.alt_label else ""
        self._norm_code = normalize_vietnamese(self.code)
        all_text = f"{self._norm_label} {self._norm_alt}"
        all_text_expanded = expand_synonyms(all_text)
        self._label_tokens = tokenize_meaningful(all_text_expanded)

    def add_alias(self, alias: str) -> None:
        norm = normalize_vietnamese(alias)
        self._aliases.append(norm)
        self._label_tokens = self._label_tokens | tokenize_meaningful(norm)


@dataclass
class ValueMatch:
    code: str
    label: str
    group: str
    score: float = 0.0
    alt_label: str = ""


class ValueCatalog:
    def __init__(
        self,
        param_name: str,
        entries: list[CatalogEntry],
        aliases: dict[str, str] | None = None,
    ):
        self.param_name = param_name
        self.entries = entries
        self.size = len(entries)

        self._alias_to_idx: dict[str, int] = {}
        for i, entry in enumerate(entries):
            self._alias_to_idx[entry._norm_label] = i
            if entry._norm_alt:
                self._alias_to_idx[entry._norm_alt] = i
            self._alias_to_idx[entry._norm_code] = i

        if aliases:
            for alias_text, target_code in aliases.items():
                norm_alias = normalize_vietnamese(alias_text)
                for i, entry in enumerate(entries):
                    if entry.code == target_code:
                        self._alias_to_idx[norm_alias] = i
                        entry.add_alias(alias_text)
                        break

    def match(
        self,
        query_normalized: str,
        query_tokens: set[str],
        top_k: int = 3,
    ) -> list[ValueMatch]:
        scored: list[tuple[float, CatalogEntry]] = []
        for entry in self.entries:
            score = self._score_entry(entry, query_normalized, query_tokens)
            if score > 0.0:
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        return [
            ValueMatch(code=entry.code, label=entry.label, group=entry.group,
                       score=score, alt_label=entry.alt_label)
            for score, entry in scored[:top_k]
        ]

    def get_all(self) -> list[ValueMatch]:
        return [
            ValueMatch(code=e.code, label=e.label, group=e.group, score=1.0, alt_label=e.alt_label)
            for e in self.entries
        ]

    def _score_entry(self, entry: CatalogEntry, query_norm: str, query_tokens: set[str]) -> float:
        code_lower = entry.code.lower()
        if len(code_lower) >= 2 and code_lower in query_norm:
            specificity = min(1.0, len(entry.code) / 5.0)
            return 0.85 + 0.15 * specificity

        if entry._norm_label and entry._norm_label in query_norm:
            return 0.95
        if entry._norm_alt and entry._norm_alt in query_norm:
            return 0.90
        for alias in entry._aliases:
            if alias in query_norm:
                return 0.88
        for alias_str, idx in self._alias_to_idx.items():
            if self.entries[idx] is entry and len(alias_str) >= 3:
                if alias_str in query_norm:
                    return 0.85

        if entry._label_tokens and query_tokens:
            overlap = query_tokens & entry._label_tokens
            if overlap:
                coverage = len(overlap) / len(entry._label_tokens)
                return min(0.70, 0.25 + 0.45 * coverage)

        if len(entry.code) <= 5 and len(entry.code) >= 2:
            code_ngrams = build_ngrams(entry._norm_code, 2)
            query_ngrams = build_ngrams(query_norm, 2)
            if code_ngrams and query_ngrams:
                jaccard = len(code_ngrams & query_ngrams) / len(code_ngrams | query_ngrams)
                if jaccard > 0.5:
                    return 0.20 + 0.30 * jaccard

        return 0.0


def load_catalog_from_json(
    json_path: str,
    aliases_path: str | None = None,
) -> dict[str, ValueCatalog]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    aliases_data = {}
    if aliases_path and Path(aliases_path).exists():
        with open(aliases_path, "r", encoding="utf-8") as f:
            aliases_data = json.load(f)

    catalogs: dict[str, ValueCatalog] = {}
    for param_name, entries_raw in raw.items():
        entries = [
            CatalogEntry(code=e.get("code", ""), label=e.get("label", ""),
                         group=e.get("group", ""), alt_label=e.get("alt_label", ""))
            for e in entries_raw
        ]
        param_aliases = aliases_data.get(param_name, {})
        catalogs[param_name] = ValueCatalog(param_name=param_name, entries=entries, aliases=param_aliases)

    return catalogs
