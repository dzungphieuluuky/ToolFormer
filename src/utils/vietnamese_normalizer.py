import re
import unicodedata
from functools import lru_cache

_VN_CHAR_MAP = {
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "ơ": "o", "Ơ": "O",
    "ư": "u", "Ư": "U",
}

_MULTI_SPACE = re.compile(r"\s+")

_VN_SYNONYMS = {
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

_VN_STOPWORDS = frozenset({
    "cua", "va", "la", "o", "tai", "cho", "voi", "trong",
    "tren", "duoi", "den", "tu", "bang", "theo", "ve",
    "nhung", "cac", "mot", "hai", "ba", "nam", "thang",
    "ngay", "toi", "can", "xem", "lay", "tim",
})


@lru_cache(maxsize=8192)
def normalize_vietnamese(text: str) -> str:
    text = text.lower()
    for src, dst in _VN_CHAR_MAP.items():
        text = text.replace(src, dst)
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^\w\s]", " ", stripped)
    result = _MULTI_SPACE.sub(" ", cleaned).strip()
    return result


def expand_synonyms(text_normalized: str) -> str:
    tokens = text_normalized.split()
    expanded = []
    for t in tokens:
        if t in _VN_SYNONYMS:
            expanded.append(_VN_SYNONYMS[t])
        else:
            expanded.append(t)
    return " ".join(expanded)


def tokenize_meaningful(text_normalized: str, min_len: int = 2) -> set[str]:
    tokens = text_normalized.split()
    return {t for t in tokens if len(t) >= min_len and t not in _VN_STOPWORDS}


def build_ngrams(text: str, n: int = 2) -> set[str]:
    text = text.replace(" ", "")
    if len(text) < n:
        return {text}
    return {text[i: i + n] for i in range(len(text) - n + 1)}
