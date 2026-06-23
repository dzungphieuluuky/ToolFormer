"""
value_catalog.py
─────────────────
Argument value catalog with alias-based lookup.

Strategy:
  - Small enums (≤ threshold): include ALL values, let LLM choose
  - Large catalogs (provinces, KPIs): normalized alias lookup + token overlap
  - Codes that appear literally in query: exact match (highest priority)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.vietnamese_normalizer import (
    normalize_vietnamese,
    expand_synonyms,
    tokenize_meaningful,
    build_ngrams,
)


@dataclass
class CatalogEntry:
    """One possible value for a parameter."""

    code: str
    label: str
    group: str = ""
    alt_label: str = ""
    # Pre-computed normalized forms (built once at load time)
    _norm_label: str = field(init=False, repr=False, default="")
    _norm_alt: str = field(init=False, repr=False, default="")
    _norm_code: str = field(init=False, repr=False, default="")
    _label_tokens: set[str] = field(init=False, repr=False, default_factory=set)
    _aliases: list[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        self._norm_label = normalize_vietnamese(self.label)
        self._norm_alt = (
            normalize_vietnamese(self.alt_label) if self.alt_label else ""
        )
        self._norm_code = normalize_vietnamese(self.code)
        # Combine all normalized forms for token matching
        all_text = f"{self._norm_label} {self._norm_alt}"
        all_text_expanded = expand_synonyms(all_text)
        self._label_tokens = tokenize_meaningful(all_text_expanded)

    def add_alias(self, alias: str) -> None:
        """Add an extra alias (e.g., from a hand-curated alias table)."""
        norm = normalize_vietnamese(alias)
        self._aliases.append(norm)
        self._label_tokens = self._label_tokens | tokenize_meaningful(norm)


@dataclass
class ValueMatch:
    """Result of value matching — serializable for JSON output."""

    code: str
    label: str
    group: str
    score: float = 0.0
    alt_label: str = ""


class ValueCatalog:
    """
    Manages argument value lookup for one parameter type.

    Matching priority:
      1. Exact code match in query (score=1.0 range)
      2. Full normalized label match in query (score=0.95)
      3. Alias match (score=0.90)
      4. Token overlap (score=0.3–0.7 based on overlap fraction)
      5. Character n-gram similarity for short codes (score=0.2–0.5)
    """

    def __init__(
        self,
        param_name: str,
        entries: list[CatalogEntry],
        aliases: dict[str, str] | None = None,
    ):
        self.param_name = param_name
        self.entries = entries
        self.size = len(entries)

        # Build reverse alias map: normalized_alias → entry index
        self._alias_to_idx: dict[str, int] = {}
        for i, entry in enumerate(entries):
            # Index by normalized label
            self._alias_to_idx[entry._norm_label] = i
            if entry._norm_alt:
                self._alias_to_idx[entry._norm_alt] = i
            # Index by normalized code
            self._alias_to_idx[entry._norm_code] = i

        # Apply external alias table
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
        """
        Score all entries against the query. Returns top-k matches.
        """
        scored: list[tuple[float, CatalogEntry]] = []

        for entry in self.entries:
            score = self._score_entry(entry, query_normalized, query_tokens)
            if score > 0.0:
                scored.append((score, entry))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        return [
            ValueMatch(
                code=entry.code,
                label=entry.label,
                group=entry.group,
                score=score,
                alt_label=entry.alt_label,
            )
            for score, entry in scored[:top_k]
        ]

    def get_all(self) -> list[ValueMatch]:
        """Return ALL values (for small enums)."""
        return [
            ValueMatch(
                code=e.code,
                label=e.label,
                group=e.group,
                score=1.0,
                alt_label=e.alt_label,
            )
            for e in self.entries
        ]

    def _score_entry(
        self,
        entry: CatalogEntry,
        query_norm: str,
        query_tokens: set[str],
    ) -> float:
        # ── Priority 1: Exact code in query ──────────────────────────
        # "5G" in "tim toc do 5g viettel" → exact hit
        code_lower = entry.code.lower()
        if len(code_lower) >= 2 and code_lower in query_norm:
            # Longer codes get higher confidence
            specificity = min(1.0, len(entry.code) / 5.0)
            return 0.85 + 0.15 * specificity

        # ── Priority 2: Full label substring in query ────────────────
        # "ha noi" in "tim toc do tai ha noi" → full match
        if entry._norm_label and entry._norm_label in query_norm:
            return 0.95

        # ── Priority 3: Full alt_label substring in query ────────────
        if entry._norm_alt and entry._norm_alt in query_norm:
            return 0.90

        # ── Priority 4: Alias match ─────────────────────────────────
        for alias in entry._aliases:
            if alias in query_norm:
                return 0.88

        # ── Priority 5: Alias table reverse lookup ───────────────────
        for alias_str, idx in self._alias_to_idx.items():
            if self.entries[idx] is entry and len(alias_str) >= 3:
                if alias_str in query_norm:
                    return 0.85

        # ── Priority 6: Token overlap ────────────────────────────────
        if entry._label_tokens and query_tokens:
            overlap = query_tokens & entry._label_tokens
            if overlap:
                # Score based on what fraction of label tokens are matched
                coverage = len(overlap) / len(entry._label_tokens)
                # Also consider how specific the overlapping tokens are
                return min(0.70, 0.25 + 0.45 * coverage)

        # ── Priority 7: Character bigram similarity (short codes) ────
        if len(entry.code) <= 5 and len(entry.code) >= 2:
            code_ngrams = build_ngrams(entry._norm_code, 2)
            query_ngrams = build_ngrams(query_norm, 2)
            if code_ngrams and query_ngrams:
                jaccard = len(code_ngrams & query_ngrams) / len(
                    code_ngrams | query_ngrams
                )
                if jaccard > 0.5:
                    return 0.20 + 0.30 * jaccard

        return 0.0


def load_catalog_from_json(
    json_path: str,
    aliases_path: str | None = None,
) -> dict[str, ValueCatalog]:
    """
    Load argument value catalogs from JSON.

    Expected format:
    {
      "location_code": [
        {"code": "HNI", "label": "Hà Nội", "group": "province", "alt_label": "Ha Noi"},
        ...
      ],
      "tech_type": [
        {"code": "5G", "label": "5G NR", "group": "technology"},
        ...
      ]
    }

    Optional aliases file (JSON):
    {
      "location_code": {
        "Sài Gòn": "HCM",
        "TPHCM": "HCM",
        "Thủ đô": "HNI"
      }
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    aliases_data: dict[str, dict[str, str]] = {}
    if aliases_path and Path(aliases_path).exists():
        with open(aliases_path, "r", encoding="utf-8") as f:
            aliases_data = json.load(f)

    catalogs: dict[str, ValueCatalog] = {}
    for param_name, entries_raw in raw.items():
        entries = [
            CatalogEntry(
                code=e.get("code", ""),
                label=e.get("label", ""),
                group=e.get("group", ""),
                alt_label=e.get("alt_label", ""),
            )
            for e in entries_raw
        ]
        param_aliases = aliases_data.get(param_name, {})
        catalogs[param_name] = ValueCatalog(
            param_name=param_name,
            entries=entries,
            aliases=param_aliases,
        )

    return catalogs


def raw_dict_to_catalogs(raw: dict[str, list[dict[str, Any]]]) -> dict[str, ValueCatalog]:
    """
    Convert a raw argument_values dict (loaded from JSON) into ValueCatalog objects.

    This is used when prepare_data.py loads argument_values.json directly
    without having a separate aliases file.
    """
    catalogs: dict[str, ValueCatalog] = {}
    for param_name, entries_raw in raw.items():
        entries = [
            CatalogEntry(
                code=e.get("code", ""),
                label=e.get("label", ""),
                group=e.get("group", ""),
                alt_label=e.get("alt_label", ""),
            )
            for e in entries_raw
        ]
        catalogs[param_name] = ValueCatalog(
            param_name=param_name,
            entries=entries,
        )
    return catalogs
