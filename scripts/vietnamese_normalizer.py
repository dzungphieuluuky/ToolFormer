"""
vietnamese_normalizer.py
────────────────────────
Vietnamese text normalization for cross-script matching.

Handles the core problem:
  Query (diacritics):  "Hà Nội"  "Đà Nẵng"  "tốc độ"
  Catalog (stripped):  "Ha Noi"  "Da Nang"   "toc do"
  Codes:               "HNI"     "DNG"        --
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


# ── Vietnamese-specific character mappings ────────────────────────────
# unicodedata.normalize("NFKD") strips MOST diacritics, but misses đ/Đ
# and some Vietnamese-specific composed characters.

_VN_CHAR_MAP = {
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    # belt-and-suspenders for rare encodings
    "ơ": "o",
    "Ơ": "O",
    "ư": "u",
    "Ư": "U",
}

# Pre-compiled pattern for whitespace collapse
_MULTI_SPACE = re.compile(r"\s+")

# Common Vietnamese abbreviations and synonyms
_VN_SYNONYMS: dict[str, str] = {
    "tp": "thanh pho",
    "tp.": "thanh pho",
    "t.p": "thanh pho",
    "tx": "thi xa",
    "tt": "thi tran",
    "q.": "quan",
    "p.": "phuong",
    "h.": "huyen",
    "brvt": "ba ria vung tau",
    "tphcm": "thanh pho ho chi minh",
    "hcm": "ho chi minh",
    "hn": "ha noi",
    "sg": "sai gon",
    "dn": "da nang",
}

# Common Vietnamese stopwords (low-value for matching)
_VN_STOPWORDS = frozenset({
    "cua",
    "va",
    "la",
    "o",
    "tai",
    "cho",
    "voi",
    "trong",
    "tren",
    "duoi",
    "den",
    "tu",
    "bang",
    "theo",
    "ve",
    "nhung",
    "cac",
    "mot",
    "hai",
    "ba",
    "nam",
    "thang",
    "ngay",
    "toi",
    "can",
    "xem",
    "lay",
    "tim",
})


@lru_cache(maxsize=8192)
def normalize_vietnamese(text: str) -> str:
    """
    Full normalization pipeline:
      1. Lowercase
      2. Vietnamese-specific char map (đ → d)
      3. NFKD decomposition (strips combining diacritics: ắ → a)
      4. Remove remaining combining characters
      5. Remove non-alphanumeric except spaces
      6. Collapse whitespace
      7. Strip

    "Hà Nội"   → "ha noi"
    "Đà Nẵng"  → "da nang"
    "Thành phố Hồ Chí Minh" → "thanh pho ho chi minh"
    """
    text = text.lower()

    # Apply Vietnamese-specific mappings BEFORE NFKD
    for src, dst in _VN_CHAR_MAP.items():
        text = text.replace(src, dst)

    # NFKD decomposition: splits composed characters into base + combining
    nfkd = unicodedata.normalize("NFKD", text)

    # Remove combining characters (diacritical marks)
    stripped = "".join(
        c for c in nfkd if not unicodedata.combining(c)
    )

    # Remove non-alphanumeric except spaces
    cleaned = re.sub(r"[^\w\s]", " ", stripped)

    # Collapse whitespace
    result = _MULTI_SPACE.sub(" ", cleaned).strip()

    return result


def expand_synonyms(text_normalized: str) -> str:
    """
    Expand Vietnamese abbreviations in already-normalized text.
    "tp hcm" → "thanh pho ho chi minh"
    """
    tokens = text_normalized.split()
    expanded: list[str] = []
    for t in tokens:
        if t in _VN_SYNONYMS:
            expanded.append(_VN_SYNONYMS[t])
        else:
            expanded.append(t)
    return " ".join(expanded)


def tokenize_meaningful(text_normalized: str, min_len: int = 2) -> set[str]:
    """
    Extract meaningful tokens (skip stopwords and very short tokens).
    """
    tokens = text_normalized.split()
    return {
        t for t in tokens if len(t) >= min_len and t not in _VN_STOPWORDS
    }


def build_ngrams(text: str, n: int = 2) -> set[str]:
    """Character n-grams for fuzzy matching on short codes."""
    text = text.replace(" ", "")
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}
