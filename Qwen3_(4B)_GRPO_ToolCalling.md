---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region _cell_guid="37fe7ef6-86a0-4e9f-9a43-4d386eee5626" _uuid="daf3acd5-9024-4e34-9d51-a3e5349926ac" jupyter={"outputs_hidden": false} -->
# Nemotron finetuning pipeline
<!-- #endregion -->

```python _cell_guid="2c096b49-0319-4946-aa1e-bba5869a1b8e" _uuid="40e91ca8-94c2-4e0f-9ee9-61a5f6682c62" jupyter={"outputs_hidden": false}
#@title Environment-Aware Installation
import os
import sys
from IPython import get_ipython

# ── Detect environment ────────────────────────────────────────────────
ENV_NAME = None
BASE_DATA_PATH = None
BASE_OUTPUT_PATH = None
DATA_MOUNT = None
KAGGLE_WHEEL_DIR = None

!export TORCH_CUDA_ARCH_LIST="12.0"
!export CMAKE_CUDA_ARCHITECTURES="120"
try:
    if "google.colab" in str(get_ipython()):
        ENV_NAME = "colab"
        BASE_DATA_PATH = "/content/"
        BASE_OUTPUT_PATH = "/content/"
        DATA_MOUNT = "/content/ToolFormer/data/generated/v1.0_k5"
        print("Environment: Google Colab")
        print("Will use uv pip install (internet available)")
    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        ENV_NAME = "kaggle"
        BASE_DATA_PATH = "/kaggle/input/"
        BASE_OUTPUT_PATH = "/kaggle/working/"
        DATA_MOUNT = "/kaggle/input/datasets/dzung271828/toolformer-data/generated/v1.0_k5"
        KAGGLE_WHEEL_DIR = "/kaggle/input/datasets/dzung271828/telco-wheels/telco-wheels/telco-wheels"
        print("Environment: Kaggle")
        print("Will use pip --no-index --find-links (offline mode)")
        # Prevent HF from trying to download
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        ENV_NAME = "local"
        BASE_DATA_PATH = "./data/"
        BASE_OUTPUT_PATH = "./output/"
        DATA_MOUNT = "data/generated/v1.0_k5"
        print("Environment: Local")
except NameError:
    ENV_NAME = "local"
    BASE_DATA_PATH = "./data/"
    BASE_OUTPUT_PATH = "./output/"
    DATA_MOUNT = "data/generated/v1.0_k5"
    print("Non-interactive session. Using local paths.")

os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
print(f"Environment: {ENV_NAME}")
print(f"Base data path: {BASE_DATA_PATH}")
print(f"Base output path: {BASE_OUTPUT_PATH}")
print(f"Data mount: {DATA_MOUNT}")
if KAGGLE_WHEEL_DIR:
    print(f"Kaggle wheel dir: {KAGGLE_WHEEL_DIR}")
    
# ── Detect GPU type ──────────────────────────────────────────────────
import subprocess as _sp
IS_T4_GPU = False
try:
    _gpu_name = (
        _sp.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .split("\n")[0]
    )
    IS_T4_GPU = "Tesla T4" in _gpu_name
    print(f"GPU type: {_gpu_name} → {'T4 (internet install)' if IS_T4_GPU else 'non-T4 (offline install)'}")
except Exception:
    print("GPU detection: nvidia-smi unavailable → defaulting to offline install")
print(f"IS_T4_GPU: {IS_T4_GPU}")
```

```python
# ── Install packages ────────────────────────────────────────────────
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"  # Disable vLLM standby mode for training SFT

if ENV_NAME == "colab":
    print("Installing packages for colab env...")
    # Colab: full internet available, use uv pip
    !pip install --upgrade -qqq uv
    try:
        import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
    except:
        _numpy = "numpy"
        _pil = "pillow"
    try:
        import subprocess
        is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except:
        is_t4 = False
    _vllm, _triton = ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm==0.15.1", "triton")
    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq {_triton}
    !uv pip install -qqq --no-deps --upgrade "torchao>=0.16.0"
    !uv pip install transformers==4.56.2
    !uv pip install --no-deps trl==0.22.2
elif ENV_NAME == "kaggle":
    print("Installing packages for kaggle env...")
    # Kaggle: no internet, use pre-loaded wheels and datasets
    import subprocess

    subprocess.run(
        "pip install -q --no-index --find-links /kaggle/input/datasets/mayukh18/nemotron-packages/packages "
        "unsloth trl peft transformers datasets accelerate bitsandbytes vllm",
        shell=True,
        check=True,
    )
    subprocess.run(
        "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        shell=True,
        check=True,
    )
    subprocess.run(
        "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        shell=True,
        check=True,
    )
    for _wd in ["/kaggle/input/datasets/llkh0a/rtx-wheels/wheels", "/kaggle/input/datasets/dzung271828/flashinfer/flashinfer-wheels"]:
        if os.path.isdir(_wd):
            subprocess.run(
                [
                    "pip",
                    "install",
                    "-q",
                    "--no-index",
                    "--find-links",
                    _wd,
                    "protobuf==6.33.5",
                    "sentencepiece",
                    "safetensors",
                    "huggingface_hub",
                    "vllm",
                ],
                check=False,
            )
    subprocess.run("rm -rf /kaggle/tmp/*", shell=True, check=True)
else:
    print("Installing packages for local env...")
    # Local: regular pip
    !pip install unsloth vllm
    !pip install transformers==4.56.2
    !pip install trl==0.22.2

print("Core stack installation complete.")
```

```python _cell_guid="deab93a3-f8f4-4b1a-a5fe-d26234b06331" _uuid="005bffe7-30b7-4031-a3f5-f21dc0d7d7a5" jupyter={"outputs_hidden": false}
#@title Install utility packages
import os
from IPython import get_ipython

_pkgs = "jsonlines openpyxl pandas tabulate seaborn matplotlib tenacity tqdm rank-bm25"

if IS_T4_GPU:
    !uv pip install {_pkgs}
else:
    !pip install --no-index --find-links=/kaggle/input/datasets/dzung271828/telco-wheels/telco-wheels/telco-wheels {_pkgs}

# Kaggle: install pre-downloaded torch + flash-attn + einops (Blackwell wheels)
if ENV_NAME == "kaggle":
    !pip install --no-index --find-links=/kaggle/input/datasets/dzung271828/telco-wheels/rtx6000-flash-attn-wheels/rtx6000-flash-attn-wheels torch flash-attn einops

```

```python _cell_guid="a955182c-dba8-4e85-ac6c-af0cc52bec87" _uuid="bd6e4b23-caa2-4376-8b6b-3e92ad4d3776" jupyter={"outputs_hidden": false}
import os
import sys
from IPython import get_ipython

def load_secret(key_name: str) -> str | None:
    """Load a secret from environment-specific secret stores.

    Args:
        key_name: Name of the secret key to load.

    Returns:
        The secret value if found, otherwise None.
    """
    env = ENV_NAME
    secret_value = None
    print(f"Attempting to load secret '{key_name}' from '{env}' environment...")
    try:
        if env == "colab":
            from google.colab import userdata

            secret_value = userdata.get(key_name)
        elif env == "kaggle":
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            secret_value = user_secrets.get_secret(key_name)
        else:
            secret_value = os.getenv(key_name)
        if not secret_value:
            print(f"Secret '{key_name}' not found in the {env} environment.")
            return None
        print(f"Successfully loaded secret '{key_name}'.")
        return secret_value
    except Exception as e:
        print(f"An error occurred while loading secret '{key_name}': {e}")
        return None


def print_system_info():
    """Print Python version, PyTorch/CUDA info, GPU count and nvidia-smi output."""
    print("\n🔧 System Information")
    print(f"Python version: {sys.version.split()[0]}")
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA not available")
    except ImportError:
        print("PyTorch not installed")
    finally:
        !nvidia-smi

is_kaggle = ENV_NAME == "kaggle"
is_colab = ENV_NAME == "colab"
is_local = ENV_NAME == "local"
print_system_info()

if not is_kaggle:
    os.environ["WANDB_API_KEY"] = wandb_key = load_secret("WANDB_API_KEY")
    os.environ["HF_TOKEN"] = HF_TOKEN = load_secret("HF_TOKEN")
    GITHUB_TOKEN = load_secret("GITHUB_TOKEN")

```

```python _cell_guid="b925d264-a345-4bd6-a61a-25aca5734269" _uuid="d1c0fb9b-2d4b-48cf-8f1c-cbe9e7a7ea30" jupyter={"outputs_hidden": false}
!find /usr -name "libcuda.so*" 2>/dev/null
```

```python _cell_guid="6505d5c0-7f1b-4809-b1de-ed2e2c3f573a" _uuid="e9dcdfef-84ee-4418-be74-7817c5d7a2a9" jupyter={"outputs_hidden": false}
import os

# Define the directory where libcuda.so lives
nvidia_lib_dir = "/usr/local/nvidia/lib64"

# Export for the compiler linker (fixes -lcuda error)
os.environ["LIBRARY_PATH"] = nvidia_lib_dir + ":" + os.environ.get("LIBRARY_PATH", "")

# Export for the runtime dynamic linker
os.environ["LD_LIBRARY_PATH"] = nvidia_lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

print("Environment paths for libcuda successfully configured!")
print(os.environ["LIBRARY_PATH"])
print(os.environ["LD_LIBRARY_PATH"])
!rm -rf ~/.cache/torch_extensions/
```

```python _cell_guid="72f73cb9-fca4-4074-b731-08fbb2952cbc" _uuid="0dae20f6-7620-41aa-a8e1-ec738015e00f" jupyter={"outputs_hidden": false}
# Clone repo (Colab) or use Kaggle dataset mount (Kaggle)
if IS_T4_GPU:
    %cd {BASE_OUTPUT_PATH}
    !rm -rf ToolFormer
    !git clone https://{GITHUB_TOKEN}@github.com/dzungphieuluuky/ToolFormer.git
    %cd {BASE_OUTPUT_PATH}/ToolFormer
else:
    print(f"Kaggle mode: data loaded from {DATA_MOUNT}")
    print("Skipping git clone (no internet). Ensure toolformer-data dataset is attached.")
    # Data lives at /kaggle/input/datasets/dzung271828/toolformer-data/

```

```python
import os
import sys
import json
import re
import math
import random
import time
import uuid
import warnings
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import unsloth
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from tabulate import tabulate

# Transformers / TRL / Unsloth
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from vllm import SamplingParams

# Datasets
try:
    from datasets import Dataset
except ImportError:
    Dataset = None

# JSON lines
try:
    import jsonlines
except ImportError:
    jsonlines = None

# Tenacity for retries
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# Rich logging
from rich.logging import RichHandler

# Set warnings
warnings.filterwarnings("ignore")

# Set random seeds
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("All imports successful.")
```

```python
# ===================== logging_utils.py =====================
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a RichHandler-based logger.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_time=True, markup=True)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger

# ===================== model_utils.py =====================
from transformers import AutoTokenizer
from unsloth import FastLanguageModel



def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    do_sample: bool = True,
) -> str:
    """Generate a response from the model for a single prompt.

    Args:
        model: The loaded model.
        tokenizer: The model tokenizer.
        prompt: Input prompt string.
        max_new_tokens: Maximum tokens to generate (default: 512).
        temperature: Sampling temperature (default: 0.6).
        do_sample: Whether to use sampling (default: True).

    Returns:
        Decoded string of generated tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)
```

```python
# ===================== sandbox.py (FIXED) =====================
# BUG FIXED: `execute_all` and `_resolve_call` imported from a nonexistent
# package `src.reward.base_reward`. This is a single-file script — those
# functions (`extract_call`, `extract_all_calls`) are already defined above
# in this same file (base_reward.py section). The bogus imports caused a
# ModuleNotFoundError at call time.

from typing import Any, Callable


def _default_mock(function_name: str, arguments: dict) -> dict:
    """Create a default mock result for a tool call.

    Args:
        function_name: Name of the mocked function.
        arguments: Arguments passed to the function.

    Returns:
        Dict with status, function name and placeholder result.
    """
    return {
        "status": "success",
        "function": function_name,
        "result": f"Mock result for {function_name}({arguments})",
    }


class Sandbox:
    """
    Executes (mock) tool calls and tracks cumulative environment state
    so that R_state can compare final_state vs gold_state.
    """

    def __init__(
        self,
        function_library: dict,
        mocks: dict[str, Callable] | None = None,
        timeout_seconds: float = 5.0,
    ):
        """Initialize Sandbox with function library and optional mocks.

        Args:
            function_library: Dict mapping function names to schemas.
            mocks: Optional dict of mock callables per function name.
            timeout_seconds: Timeout for mock execution in seconds.
        """
        self.library = function_library
        self.mocks = mocks or {}
        self.timeout = timeout_seconds
        self._call_log: list[dict] = []
        self._state: dict = {}

    # ── public API ───────────────────────────────────────────────────────
    def execute(self, call_input: str | dict | None) -> bool:
        """Execute a single tool call.

        Args:
            call_input: A tool call as a string, dict, or None.

        Returns:
            True if the call succeeded, False otherwise.
        """
        if call_input is None:
            return False
        call = self._resolve_call(call_input)
        if call is None:
            return False
        return self._run_call(call)

    def execute_all(self, response: str) -> list[bool]:
        """Parse and execute all tool calls from a model response.

        Args:
            response: Model output containing tool calls.

        Returns:
            List of booleans indicating success per call.
        """
        # FIXED: extract_all_calls is defined locally in this file (base_reward.py section)
        calls = extract_all_calls(response)
        return [self._run_call(c) for c in calls]

    def get_call_log(self) -> list[dict]:
        """Return a copy of the call log."""
        return self._call_log.copy()

    def get_state(self) -> dict:
        """Return the cumulative environment state snapshot (for R_state)."""
        return self._state.copy()

    def clear(self) -> None:
        """Reset both call log and environment state."""
        self._call_log.clear()
        self._state.clear()

    def clear_log(self) -> None:
        """Clear both call log and environment state (delegates to clear())."""
        self.clear()

    # ── internals ────────────────────────────────────────────────────────
    def _resolve_call(self, call_input: str | dict) -> dict | None:
        """Parse call input into a standard dict format.

        Args:
            call_input: Raw call input as string or dict.

        Returns:
            Parsed call dict, or None if parsing failed.
        """
        if isinstance(call_input, dict):
            return call_input
        # FIXED: extract_call is defined locally in this file (base_reward.py section)
        return extract_call(call_input)

    def _run_call(self, call: dict) -> bool:
        """Execute a parsed call: validate schema, run mock, log result.

        Args:
            call: Parsed call dict with "function" and "arguments" keys.

        Returns:
            True if the call succeeded, False otherwise.
        """
        func_name = call.get("function")
        arguments = call.get("arguments", {})
        if not func_name:
            return False
        if func_name not in self.library:
            self._log(func_name, arguments, "error", "function_not_found")
            return False
        schema = self.library[func_name]
        if not self._validate_args(func_name, arguments, schema):
            return False
        try:
            mock_fn = self.mocks.get(func_name, _default_mock)
            if mock_fn is _default_mock:
                result = _default_mock(func_name, arguments)
            else:
                result = mock_fn(**arguments)
            self._log(func_name, arguments, "success", result)
            self._state[func_name] = {
                "arguments": arguments,
                "result": result,
            }
            return True
        except Exception as exc:
            self._log(func_name, arguments, "error", str(exc))
            return False

    def _validate_args(self, func_name: str, arguments: dict, schema: dict) -> bool:
        """Validate arguments against function schema.

        Checks required parameters and constraints (min, max, enum).

        Args:
            func_name: Function name for error logging.
            arguments: Arguments to validate.
            schema: Function schema dict with parameters and constraints.

        Returns:
            True if arguments pass validation, False otherwise.
        """
        params = schema.get("parameters", {})
        constraints = schema.get("constraints", {})
        for pname, pinfo in params.items():
            if isinstance(pinfo, dict) and pinfo.get("required", False) and pname not in arguments:
                self._log(func_name, arguments, "error", f"missing_required:{pname}")
                return False
        for pname, con in constraints.items():
            if pname not in arguments:
                continue
            val = arguments[pname]
            if "min" in con and val < con["min"]:
                self._log(func_name, arguments, "error",
                          f"constraint_violation:{pname}<{con['min']}")
                return False
            if "max" in con and val > con["max"]:
                self._log(func_name, arguments, "error",
                          f"constraint_violation:{pname}>{con['max']}")
                return False
            if "enum" in con and val not in con["enum"]:
                self._log(func_name, arguments, "error",
                          f"constraint_violation:{pname} not in enum")
                return False
        return True

    def _log(self, func_name: str, arguments: dict, status: str, result: Any) -> None:
        """Append an entry to the call log.

        Args:
            func_name: Name of the called function.
            arguments: Arguments passed to the function.
            status: Outcome status ("success" or "error").
            result: The call result or error message.
        """
        self._call_log.append({
            "function": func_name,
            "arguments": arguments,
            "status": status,
            "result": result,
        })


def build_gold_state(gold_calls: list[dict], function_library: dict) -> dict:
    """Replay gold tool calls through a Sandbox to produce gold state.

    Args:
        gold_calls: List of gold-standard tool call dicts.
        function_library: Dict mapping function names to schemas.

    Returns:
        Dict of the resulting environment state for R_state comparison.
    """
    sb = Sandbox(function_library)
    for call in gold_calls:
        sb.execute(call)
    return sb.get_state()
```

```python
# ===================== vietnamese_normalizer.py =====================
"""
Vietnamese text normalization for cross-script matching.

Handles the core problem:
  Query (diacritics):  "Hà Nội"  "Đà Nẵng"  "tốc độ"
  Catalog (stripped):  "Ha Noi"  "Da Nang"   "toc do"
  Codes:               "HNI"     "DNG"        --
"""

import re
import unicodedata
from functools import lru_cache


# ── Vietnamese-specific character mappings ────────────────────────────
# unicodedata.normalize("NFKD") strips MOST diacritics, but misses đ/Đ
# and some Vietnamese-specific composed characters.

_VN_CHAR_MAP = {
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    # belt-and-suspenders for rare encodings
    "ơ": "o", "Ơ": "O",
    "ư": "u", "Ư": "U",
}

# Pre-compiled pattern for whitespace collapse
_MULTI_SPACE = re.compile(r"\s+")

# Common Vietnamese abbreviations and synonyms
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

# Common Vietnamese stopwords (low-value for matching)
_VN_STOPWORDS = frozenset({
    "cua", "va", "la", "o", "tai", "cho", "voi", "trong",
    "tren", "duoi", "den", "tu", "bang", "theo", "ve",
    "nhung", "cac", "mot", "hai", "ba", "nam", "thang",
    "ngay", "toi", "can", "xem", "lay", "tim",
})


@lru_cache(maxsize=8192)
def normalize_vietnamese(text: str) -> str:
    """
    Full normalization pipeline:
      1. Lowercase
      2. NFKD decomposition (strips combining diacritics: ắ → a)
      3. Vietnamese-specific char map (đ → d)
      4. Remove remaining combining characters
      5. Collapse whitespace
      6. Strip

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
    expanded = []
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
        t for t in tokens
        if len(t) >= min_len and t not in _VN_STOPWORDS
    }


def build_ngrams(text: str, n: int = 2) -> set[str]:
    """Character n-grams for fuzzy matching on short codes."""
    text = text.replace(" ", "")
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}
```

```python
# ===================== value_catalog.py =====================
"""
Argument value catalog with alias-based lookup.

This replaces embedding-based value matching with a deterministic,
fast, and correct approach for Vietnamese telecom arguments.

Strategy:
  - Small enums (≤ threshold): include ALL values, let LLM choose
  - Large catalogs (provinces, KPIs): normalized alias lookup + token overlap
  - Codes that appear literally in query: exact match (highest priority)
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    _label_tokens: frozenset = field(init=False, repr=False, default_factory=frozenset)
    _aliases: list[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        self._norm_label = normalize_vietnamese(self.label)
        self._norm_alt = normalize_vietnamese(self.alt_label) if self.alt_label else ""
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
    """Result of value matching — compatible with retriever block builder."""
    code: str
    label: str
    group: str
    score: float = 0.0
    alt_label: str = ""


class ValueCatalog:
    """
    Manages argument value lookup for one parameter type.

    Matching priority:
      1. Exact code match in query (score=1.0)
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
            score = self._score_entry(
                entry, query_normalized, query_tokens
            )
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
        # Check if any known alias substring is in the query
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

    aliases_data = {}
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
```

```python
# ===================== smart_retriever.py =====================
"""
Smart retriever that combines:
  - Function retrieval: BM25 + optional lightweight embedding (hybrid)
  - Value retrieval: deterministic normalized lookup (NO embeddings)

Handles Vietnamese diacritics ↔ ASCII/code mapping correctly.
"""

from __future__ import annotations

import re
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────────────────────────────
# Date extraction (regex-based, no model needed)
# ─────────────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    # "tháng 6/2026", "thang 6 nam 2026"
    (
        r"(?:thang|tháng)\s*(\d{1,2})\s*[/\-\.]\s*(\d{4})",
        lambda m: (f"{m.group(2)}-{int(m.group(1)):02d}-01",
                   _last_day(int(m.group(2)), int(m.group(1)))),
    ),
    # "tháng 6 năm 2026"
    (
        r"(?:thang|tháng)\s*(\d{1,2})\s*(?:nam|năm)\s*(\d{4})",
        lambda m: (f"{m.group(2)}-{int(m.group(1)):02d}-01",
                   _last_day(int(m.group(2)), int(m.group(1)))),
    ),
    # "năm 2022", "nam 2022"
    (
        r"(?:toan\s*)?(?:nam|năm)\s*(\d{4})",
        lambda m: (f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"),
    ),
    # "Q1/2025", "quý 2 năm 2025"
    (
        r"(?:q|quy|quý)\s*(\d)\s*[/\-]?\s*(\d{4})",
        lambda m: _quarter_range(int(m.group(1)), int(m.group(2))),
    ),
    # "tuần 22/2026", "thứ 5/2026" (week/ordinal — NOT month)
    (
        r"(?:tuần|tuan|thứ|thu|week)\s+\d{1,2}\s*/\s*\d{4}",
        lambda m: ("", ""),
    ),
    # "06/2026" (MM/YYYY)
    (
        r"(\d{1,2})\s*/\s*(\d{4})",
        lambda m: (f"{m.group(2)}-{int(m.group(1)):02d}-01",
                   _last_day(int(m.group(2)), int(m.group(1)))),
    ),
    # "2022-01-01" to "2022-12-31" (explicit ISO range)
    (
        r"(\d{4}-\d{2}-\d{2})\s*(?:den|đến|to|\-)\s*(\d{4}-\d{2}-\d{2})",
        lambda m: (m.group(1), m.group(2)),
    ),
]


def _last_day(year: int, month: int) -> str:
    """Last day of a given month."""
    import calendar
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last:02d}"


def _quarter_range(q: int, year: int) -> tuple[str, str]:
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return (f"{year}-{starts[q]}", f"{year}-{ends[q]}")


def extract_dates(query: str) -> dict[str, str]:
    """
    Extract from_date and to_date from Vietnamese query.
    Returns dict with 'from_date' and/or 'to_date' keys.
    """
    query_lower = query.lower()
    for pattern, extractor in _DATE_PATTERNS:
        for match in re.finditer(pattern, query_lower):
            start = match.start()
            prefix = query_lower[max(0, start - 8):start].strip()
            # Skip matches preceded by week/ordinal indicators
            if re.search(r"(?:tuần|tuan|thứ|thu|week)\s*$", prefix):
                continue
            try:
                from_date, to_date = extractor(match)
                return {"from_date": from_date, "to_date": to_date}
            except (ValueError, IndexError, calendar.IllegalMonthError):
                continue
    return {}


# ─────────────────────────────────────────────────────────────────────
# Data-level extraction (regex-based)
# ─────────────────────────────────────────────────────────────────────

_DATA_LEVEL_PATTERNS = [
    (r"(?:theo|tong\s*hop)\s*(?:ngay|ngày|daily)", "day"),
    (r"(?:theo|tong\s*hop)\s*(?:tuan|tuần|weekly)", "week"),
    (r"(?:theo|tong\s*hop)\s*(?:thang|tháng|monthly)", "month"),
    (r"(?:theo|tong\s*hop)\s*(?:nam|năm|yearly)", "year"),
    (r"(?:hang|hàng)\s*(?:ngay|ngày)", "day"),
    (r"(?:hang|hàng)\s*(?:tuan|tuần)", "week"),
    (r"(?:hang|hàng)\s*(?:thang|tháng)", "month"),
    # Implicit from date range
    (r"(?:thang|tháng)\s*\d{1,2}", "month"),  # "tháng 6" implies monthly context
]


def extract_data_level(query: str) -> str | None:
    """Extract aggregation level from Vietnamese query."""
    query_norm = normalize_vietnamese(query)
    for pattern, level in _DATA_LEVEL_PATTERNS:
        if re.search(pattern, query_norm):
            return level
    return None


# ─────────────────────────────────────────────────────────────────────
# Function retriever: BM25 over normalized Vietnamese descriptions
# ─────────────────────────────────────────────────────────────────────

class FunctionRetriever:
    """
    BM25-based function retrieval with Vietnamese normalization.

    For most telecom function libraries (10–200 functions), BM25 with
    proper normalization is sufficient. Embeddings add marginal value
    but significant latency.

    If you need embeddings, pass method="hybrid" and an encoder_model
    that handles Vietnamese (e.g., "keepitreal/vietnamese-sbert").
    """

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

        # Build search corpus — normalize everything
        self._search_texts = []
        self._search_tokens = []
        for name, schema in function_library.items():
            text = self._build_search_text(name, schema)
            norm = normalize_vietnamese(text)
            expanded = expand_synonyms(norm)
            self._search_texts.append(expanded)
            self._search_tokens.append(expanded.split())

        self._bm25 = BM25Okapi(self._search_tokens)

        # Optional embedding
        self._encoder = None
        self._embeddings = None
        if method == "hybrid" and encoder_model:
            self._init_embeddings(encoder_model)

    @staticmethod
    def _build_search_text(name: str, schema: dict) -> str:
        """Build a rich searchable string from function schema."""
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
            self._search_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
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
        emb = self._minmax(self._emb_scores(query))  # raw query for embedding
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


# ─────────────────────────────────────────────────────────────────────
# Argument value retriever: deterministic lookup + scoring
# ─────────────────────────────────────────────────────────────────────

class ArgumentValueRetriever:
    """
    Retrieves relevant argument values using deterministic normalized
    string matching. NO embedding model needed.

    Strategy:
      1. If param has ≤ include_all_threshold values → return ALL
      2. Otherwise → normalized alias lookup + token overlap scoring
      3. Dates → regex extraction
      4. data_level → regex extraction
    """

    def __init__(
        self,
        catalogs: dict[str, ValueCatalog],
        top_k_values: int = 5,
        include_all_threshold: int = 12,
    ):
        self.catalogs = catalogs
        self.top_k = top_k_values
        self.include_all_threshold = include_all_threshold

        # Build param name → catalog key mapping
        # Handles cases where schema param name doesn't exactly match catalog key
        self._param_to_catalog: dict[str, str] = {}
        for key in catalogs:
            self._param_to_catalog[key] = key

    def add_param_mapping(self, param_name: str, catalog_key: str) -> None:
        """Explicitly map a schema parameter name to a catalog key."""
        self._param_to_catalog[param_name] = catalog_key

    def retrieve_for_function(
        self,
        query: str,
        function_schema: dict,
    ) -> dict[str, list[ValueMatch]]:
        """
        For each parameter in the function schema, find matching values.
        """
        result: dict[str, list[ValueMatch]] = {}
        params = function_schema.get("parameters", {})

        # Pre-normalize query once
        query_norm = expand_synonyms(normalize_vietnamese(query))
        query_tokens = tokenize_meaningful(query_norm)

        for param_name in params:
            # ── Special handling: dates ───────────────────────────────
            if param_name in ("from_date", "to_date"):
                dates = extract_dates(query)
                if param_name in dates:
                    result[param_name] = [
                        ValueMatch(
                            code=dates[param_name],
                            label=dates[param_name],
                            group="extracted_date",
                            score=1.0,
                        )
                    ]
                continue

            # ── Special handling: data_level ──────────────────────────
            if param_name == "data_level":
                level = extract_data_level(query)
                if level:
                    result[param_name] = [
                        ValueMatch(
                            code=level,
                            label=level,
                            group="extracted_level",
                            score=1.0,
                        )
                    ]
                # Also include catalog values if they exist
                catalog = self._get_catalog(param_name)
                if catalog:
                    if catalog.size <= self.include_all_threshold:
                        all_vals = catalog.get_all()
                        if param_name in result:
                            # Merge: extracted first, then catalog
                            existing_codes = {m.code for m in result[param_name]}
                            for v in all_vals:
                                if v.code not in existing_codes:
                                    result[param_name].append(v)
                        else:
                            result[param_name] = all_vals
                continue

            # ── Standard catalog lookup ───────────────────────────────
            catalog = self._get_catalog(param_name)
            if catalog is None:
                # Check schema constraints for inline enum
                constraints = function_schema.get("constraints", {})
                param_con = constraints.get(param_name, {})
                if "enum" in param_con:
                    result[param_name] = [
                        ValueMatch(
                            code=str(v), label=str(v), group="enum", score=1.0
                        )
                        for v in param_con["enum"]
                    ]
                continue

            # Small enum → include all
            if catalog.size <= self.include_all_threshold:
                result[param_name] = catalog.get_all()
            else:
                matches = catalog.match(
                    query_norm, query_tokens, top_k=self.top_k
                )
                if matches:
                    result[param_name] = matches

        return result

    def retrieve_for_functions(
        self,
        query: str,
        function_names: list[str],
        function_library: dict,
    ) -> dict[str, list[ValueMatch]]:
        """Merge value matches across multiple candidate functions."""
        combined: dict[str, list[ValueMatch]] = {}

        for fn in function_names:
            if fn not in function_library:
                continue
            fn_results = self.retrieve_for_function(
                query, function_library[fn]
            )
            for param_name, matches in fn_results.items():
                if param_name not in combined:
                    combined[param_name] = matches
                else:
                    existing_codes = {m.code for m in combined[param_name]}
                    for m in matches:
                        if m.code not in existing_codes:
                            combined[param_name].append(m)
                    combined[param_name].sort(key=lambda x: -x.score)
                    combined[param_name] = combined[param_name][: self.top_k]

        return combined

    def _get_catalog(self, param_name: str) -> ValueCatalog | None:
        """Resolve param name to catalog, with fallback matching."""
        # Direct match
        catalog_key = self._param_to_catalog.get(param_name)
        if catalog_key and catalog_key in self.catalogs:
            return self.catalogs[catalog_key]

        # Direct key match
        if param_name in self.catalogs:
            return self.catalogs[param_name]

        # Suffix match: "network_provider" matches catalog key "provider"
        best_key = None
        best_len = 0
        for key in self.catalogs:
            if param_name.endswith(key) and len(key) > best_len:
                best_key = key
                best_len = len(key)
        if best_key:
            return self.catalogs[best_key]

        # Prefix match
        for key in self.catalogs:
            if param_name.startswith(key):
                return self.catalogs[key]

        return None


# ─────────────────────────────────────────────────────────────────────
# Combined TelcoRetriever
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    function_names: list[str]
    argument_values: dict[str, list[ValueMatch]]
    extracted_dates: dict[str, str]
    extracted_data_level: str | None


class TelcoRetriever:
    """
    Combined function + argument value retriever.

    Usage:
        catalogs = load_catalog_from_json("data/argument_values.json",
                                          "data/aliases.json")
        retriever = TelcoRetriever.build(
            function_library=lib,
            catalogs=catalogs,
            method="bm25",
        )
        result = retriever.retrieve(query, function_library)
    """

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
        func_ret = FunctionRetriever(
            function_library, method=method, encoder_model=encoder_model
        )
        val_ret = ArgumentValueRetriever(
            catalogs,
            top_k_values=top_k_values,
            include_all_threshold=include_all_threshold,
        )
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

        # Also extract dates and data_level as standalone metadata
        dates = extract_dates(query)
        data_level = extract_data_level(query)

        return RetrievalResult(
            function_names=func_names,
            argument_values=arg_values,
            extracted_dates=dates,
            extracted_data_level=data_level,
        )
```

```python
# ===================== excel_parser.py =====================
def _safe_json(value: str, fallback=None):
    if not isinstance(value, str) or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except Exception:
            return fallback

def _safe_list(value: str) -> list:
    result = _safe_json(value, fallback=None)
    if isinstance(result, list):
        return result
    if isinstance(value, str):
        return [v.strip() for v in value.replace("\n", ",").split(",") if v.strip()]
    return []

def parse_telecom_functions(excel_path: str, output_path: Optional[str] = None) -> dict:
    import ast
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    df = pd.read_excel(path)
    required_cols = {"function_name", "description", "parameters"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")
    library = {}
    for idx, row in df.iterrows():
        name = str(row["function_name"]).strip()
        if not name:
            continue
        params = _safe_json(row.get("parameters", "{}"), fallback={})
        examples = _safe_list(row.get("example_queries", "[]"))
        domain = _safe_json(row.get("domain_info", "{}"), fallback={})
        constraints = _safe_json(row.get("constraints", "{}"), fallback={})
        tags = _safe_list(row.get("tags", "[]"))
        func_schema = {
            "name": name,
            "description": str(row["description"]).strip(),
            "parameters": params,
            "examples": examples,
            "domain": domain,
            "constraints": constraints,
            "tags": tags,
        }
        library[name] = func_schema
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(library, fh, indent=2, ensure_ascii=False)
        print(f"[excel_parser] Saved {len(library)} functions → {out}")
    return library

def load_function_library(library_path: str) -> dict:
    with open(library_path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def load_function_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
```

```python
# ===================== base_reward.py =====================
import re
import json
from typing import Any

# ─────────────────────────────────────────────────────────────────────
# _parse_gt: safely coerce a ground_truth column value (JSON string
# OR already-a-dict, for backward compatibility) into a dict.
#
# BUG FIXED: format_sample_for_grpo now serialises ground_truth to a
# JSON string to avoid pyarrow struct-schema inference on heterogeneous
# argument value types (ArrowInvalid). Every consumer that receives
# ground_truth from the HF Dataset must route it through this helper.
# ─────────────────────────────────────────────────────────────────────
# ===== ADD THIS =====
def normalize_ground_truth(gt: dict) -> dict:
    """Normalise a ground_truth dict to a consistent schema."""
    if not isinstance(gt, dict):
        return {}
    calls = gt.get("calls", [])
    if not isinstance(calls, list):
        calls = []
    normalized_calls = []
    for c in calls:
        if isinstance(c, dict):
            func = c.get("function")
            args = c.get("arguments", {})
            if func:
                normalized_calls.append({"function": func, "arguments": args})
    return {
        "calls": normalized_calls,
        "reasoning": gt.get("reasoning", ""),
    }
# ===== END ADD =====
def _parse_gt(gt) -> dict:
    """Coerce ground_truth (JSON string or dict) into a dict and normalize."""
    if isinstance(gt, dict):
        return normalize_ground_truth(gt)
    if isinstance(gt, str):
        try:
            parsed = json.loads(gt)
            return normalize_ground_truth(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)

# aliases
_CALL_RE = _TOOL_CALL_RE
_REASON_RE = _REASONING_RE


def extract_call(response: str) -> dict | None:
    """Extract the first <tool_call>...</tool_call> JSON from a response."""
    match = _TOOL_CALL_RE.search(response)
    if not match:
        # legacy fallback
        old_re = re.compile(r"<call>(.*?)</call>", re.DOTALL)
        match = old_re.search(response)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.lower() == "null":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # lenient: try to find the first { ... } substring
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        return None


def extract_all_calls(response: str) -> list[dict]:
    """Extract ALL <tool_call> JSON blocks from a response (for sequential workflows)."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(response):
        raw = m.group(1).strip()
        if raw.lower() == "null":
            continue
        try:
            parsed = json.loads(raw)
            if parsed is not None:
                calls.append(parsed)
        except json.JSONDecodeError:
            pass
    return calls


def extract_reasoning(response: str) -> str:
    match = _REASONING_RE.search(response)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────────────────────────────
# Component reward functions
# ─────────────────────────────────────────────────────────────────────

def schema_valid(response: str) -> float:
    return 1.0 if extract_call(response) is not None else 0.0


def func_selection_ok(response: str, expected_func: str) -> float:
    call = extract_call(response)
    if call is None:
        return 0.0
    return 1.0 if call.get("function") == expected_func else 0.0


def args_accuracy(response: str | dict, expected_args: dict) -> float:
    call = extract_call(response) if isinstance(response, str) else response
    if call is None:
        return 0.0
    if not expected_args:
        return 1.0
    actual = call.get("arguments", {})
    correct = sum(
        1
        for k, v in expected_args.items()
        if str(actual.get(k, "")).strip() == str(v).strip()
    )
    return correct / len(expected_args)


def reasoning_quality(response: str) -> float:
    text = extract_reasoning(response)
    if not text:
        return 0.0
    words = len(text.split())
    length_score = min(1.0, words / 50.0)
    has_steps = bool(
        re.search(
            r"(step\s*\d|first|then|finally|because|therefore|since)",
            text,
            re.IGNORECASE,
        )
    )
    return 0.7 * length_score + 0.3 * float(has_steps)


# ─────────────────────────────────────────────────────────────────────
# Trajectory-level reward for RC-GRPO
# Uses the SAME {function, arguments} schema as dataset & extract_call
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Composite reward for GRPO training (works for both single & sequential)
# ─────────────────────────────────────────────────────────────────────

def make_reward_function(
    ground_truth: dict,
    function_library: dict,
    sandbox_cls=None,
    weights: dict | None = None,
):
    """
    Build a reward callable(response_text) -> float for one training sample.

    Supports both single_call and sequential workflows via normalize_ground_truth.
    Returns 1.0 for correct abstention (no gold calls, no agent calls).
    """
    gt = normalize_ground_truth(ground_truth)
    w = weights or {
        "format": 0.10,
        "function": 0.30,
        "arguments": 0.40,
        "reasoning": 0.10,
        "execution": 0.10,
    }

    gold_calls = [
        {"function": c["function"], "arguments": c.get("arguments", {})}
        for c in gt.get("calls", [])
    ]
    is_abstention = len(gold_calls) == 0

    def _reward(response_text: str) -> float:
        score = 0.0

        # 1. Format reward
        has_tool_call = bool(_TOOL_CALL_RE.search(response_text))
        has_reasoning = bool(_REASONING_RE.search(response_text))
        fmt = 0.0
        if has_tool_call:
            fmt += 0.5
        if has_tool_call and has_reasoning:
            fmt += 0.5
        score += w["format"] * fmt

        # 2. Extract calls
        agent_calls = extract_all_calls(response_text)

        if not agent_calls:
            if is_abstention:
                return 1.0
            return score  # only format score

        if is_abstention:
            return 0.0

        # 3. Function selection — fraction of gold funcs matched by name
        gold_func_names = [c["function"] for c in gold_calls]
        agent_func_names = [c.get("function") for c in agent_calls]
        func_hits = sum(
            1 for gf in gold_func_names if gf in agent_func_names
        )
        score += w["function"] * (func_hits / len(gold_func_names))

        # 4. Argument accuracy — average across gold calls
        arg_scores = []
        for gc in gold_calls:
            best = 0.0
            for ac in agent_calls:
                if ac.get("function") == gc["function"]:
                    best = max(best, args_accuracy(ac, gc.get("arguments", {})))
            arg_scores.append(best)
        score += w["arguments"] * (sum(arg_scores) / len(arg_scores))

        # 5. Reasoning quality
        score += w["reasoning"] * reasoning_quality(response_text)

        # 6. Execution (sandbox)
        if sandbox_cls is not None:
            try:
                sb = sandbox_cls(function_library)
                results = sb.execute_all(response_text)
                exec_score = sum(results) / max(len(results), 1)
                score += w["execution"] * exec_score
            except Exception:
                pass

        return score

    return _reward


def make_binary_reward_function(ground_truth: dict, function_library: dict):
    """
    Build a binary reward callable(response_text) -> int (0 or 1).
    Returns 1 only if ALL gold calls are matched exactly.
    Returns 1 for correct abstention (no gold calls, no agent calls).
    Compatible with RC-GRPO's R(τ).
    """
    gt = normalize_ground_truth(ground_truth)
    gold_calls = [
        {"function": c["function"], "arguments": c.get("arguments", {})}
        for c in gt.get("calls", [])
    ]
    is_abstention = len(gold_calls) == 0

    def _reward(response_text: str) -> int:
        agent_calls = extract_all_calls(response_text)
        if not agent_calls:
            return 1 if is_abstention else 0
        if is_abstention:
            return 0
        return compute_action_coverage_reward(agent_calls, gold_calls)

    return _reward


# ─────────────────────────────────────────────────────────────────────
# TRL-compatible batch reward functions
# ─────────────────────────────────────────────────────────────────────

def function_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """TRL-compatible: measures function selection accuracy against ground truth."""
    rewards = []
    for c, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        calls = gt.get("calls", [])
        expected_func = calls[0].get("function", "") if calls else ""
        rewards.append(func_selection_ok(c, expected_func))
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for c in completions:
        has_tool_call = bool(_TOOL_CALL_RE.search(c))
        has_reasoning = bool(_REASONING_RE.search(c))
        score = 0.0
        if has_tool_call:
            score += 0.5
        if has_tool_call and has_reasoning:
            score += 0.5
        rewards.append(score)
    return rewards

def argument_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """FIXED: parses JSON-string ground_truth via _parse_gt before use."""
    """TRL-compatible: measures argument accuracy against ground truth."""
    rewards = []
    for c, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        calls = gt.get("calls", [])
        expected_args = calls[0].get("arguments", {}) if calls else {}
        rewards.append(args_accuracy(c, expected_args))
    return rewards


def composite_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """TRL-compatible: weighted sum of format + function + arguments + reasoning."""
    rewards = []
    for c, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        calls = gt.get("calls", [])
        if not calls:
            agent = extract_call(c)
            rewards.append(1.0 if agent is None else 0.0)
        else:
            first_call = calls[0]
            expected_func = first_call.get("function", "")
            expected_args = first_call.get("arguments", {})
            r = (
                0.10 * (1.0 if bool(_TOOL_CALL_RE.search(c)) and bool(_REASONING_RE.search(c)) else
                        0.5 if bool(_TOOL_CALL_RE.search(c)) else 0.0)
                + 0.30 * func_selection_ok(c, expected_func)
                + 0.40 * args_accuracy(c, expected_args)
                + 0.20 * reasoning_quality(c)
            )
            rewards.append(r)
    return rewards
```

```python
"""
RC-GRPO: Reward-Conditioned Group Relative Policy Optimization
Faithful implementation based on arXiv:2602.03025

CORRECTED VERSION — fixes:
  1. Token naming: <|high_reward|> / <|low_reward|> (underscores, matching paper)
  2. Importance ratio: π_θ / π_θ_old (NOT π_ref)
  3. Gradient flow: fresh forward pass with grad for current log-probs
  4. KL estimator: Schulman k3 (unbiased, non-negative)
  5. Action schema: {function, arguments} everywhere
  6. Token-level clipping (standard GRPO)
  7. Reference model loading (no deepcopy of quantized model)
"""

import math
import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. REWARD TOKEN CONSTANTS
#    Paper Section 3.1: "<|high_reward|> and <|low_reward|>"
# ─────────────────────────────────────────────────────────────────────────────

HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"
REWARD_TOKENS = [HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN]


def compute_action_coverage_reward(
    agent_tool_calls: List[Dict],
    gold_tool_calls: List[Dict],
) -> int:
    """
    R_action: 1 iff every gold call is covered by some agent call.
    Uses {function, arguments} keys consistently.
    """

    def _tool_call_matches(agent_call: dict, gold_call: dict) -> bool:
        if agent_call.get("function") != gold_call.get("function"):
            return False
        gold_args = gold_call.get("arguments", {})
        agent_args = agent_call.get("arguments", {})
        for param_key, param_val in gold_args.items():
            if str(agent_args.get(param_key, "")).strip() != str(param_val).strip():
                return False
        return True

    for gold_call in gold_tool_calls:
        if not any(_tool_call_matches(a, gold_call) for a in agent_tool_calls):
            return 0
    return 1


def compute_group_normalized_advantages(
    group_rewards: List[float],
    epsilon_stable: float = 1e-8,
) -> List[float]:
    """
    A_j = (R_j - mu_g) / (sigma_g + eps_stab)      (Eq. 6)

    sigma_g is the POPULATION std over the group (ddof=0), matching the
    paper exactly: sigma_g = sqrt( (1/G) * sum_k (R(tau_k) - mu_g)^2 ).
    """
    rewards = np.array(group_rewards, dtype=np.float64)
    mu_g = rewards.mean()
    sigma_g = rewards.std(ddof=0)  # FIXED: population std, matches Eq. 6 exactly
    return ((rewards - mu_g) / (sigma_g + epsilon_stable)).tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 4. REWARD-CONDITIONED ROLLOUT SAMPLING (Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

def sample_reward_token_for_group(
    group_size: int,
    high_reward_probability: float = 0.5,
) -> List[str]:
    """Sample G reward tokens from P_sample(r) (Eq. 3)."""
    return [
        HIGH_REWARD_TOKEN
        if random.random() < high_reward_probability
        else LOW_REWARD_TOKEN
        for _ in range(group_size)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. RCTP DATASET (Section 3.2 / Appendix B)
#    BUG FIXED (x2):
#      1. Paper Appendix B: "reward token appended to the FIRST user message"
#         — singular. The old code injected `[Reward Goal: ...]` into EVERY
#         user turn, which is wrong and also leaks the token into multi-turn
#         histories in a way the paper never does.
#      2. There was no function anywhere that actually BUILDS the mixed
#         expert/failure dataset D = {(tau_i, r_i)} from your
#         train_dataset.jsonl. `Trajectory`/`RCTPDataset` existed but were
#         never instantiated from real data. This fix adds
#         `build_rctp_dataset_from_jsonl(...)` which:
#           - loads expert trajectories directly from ground_truth (R=1)
#           - generates failure trajectories via exploration rollouts of the
#             (not-yet-RL-tuned) base/RCTP-in-progress policy (R=0, mismatched
#             function/argument calls)
#           - balances them to an exact 1:1 ratio (Table 6), matching the
#             paper's RCTP-FT statistics (720 success : 720 failure on BFCL).
# ─────────────────────────────────────────────────────────────────────────────

def _make_multi_call_failure_response(
    gold_calls: list[dict],
    function_library: dict,
    retriever=None,
    catalog: dict | None = None,
    retrieved_functions: list[str] | None = None,
    query: str = "",
) -> str:
    """
    Synthesize a FAILURE trajectory for multi-call samples (sequential/parallel).
    Randomly chooses one of:
      - skip one call entirely (missing step)
      - corrupt one call (wrong function / missing arg / wrong value)
    Uses retriever-aware function selection and catalog-aware value corruption.
    """
    import random as _random

    _REASONING_POOL = [
        "Analyzing the query and selecting the matching function and arguments.",
        "Evaluating the network operator's request and mapping it to the appropriate system function.",
        "Processing the Vietnamese telecom query to identify the required network management operation.",
        "Reviewing available telecom diagnostic tools and selecting the most suitable one for this scenario.",
        "Determining which network function handles the requested operation based on query parameters.",
        "Parsing the operator query to extract location, technology, and time constraints for the function call.",
    ]

    reasoning = _random.choice(_REASONING_POOL)
    failure_kind = _random.choice(["skip_call", "corrupt_one"])

    # Helper: pick a confusable function
    def _corrupt_single_call(call: dict) -> dict:
        cf = call["function"]
        ca = call.get("arguments", {})
        fk = _random.choice(["wrong_function", "missing_arg", "wrong_arg_value"])
        if fk == "wrong_function":
            wrong_func = _pick_confusable_func(cf)
            if wrong_func is None and len(function_library) > 1:
                c = [f for f in function_library if f != cf]
                wrong_func = _random.choice(c) if c else cf
            return {"function": wrong_func or cf, "arguments": ca}
        elif fk == "missing_arg" and ca:
            corrupted = dict(ca)
            drop_key = _random.choice(list(corrupted.keys()))
            corrupted.pop(drop_key)
            return {"function": cf, "arguments": corrupted}
        elif fk == "wrong_arg_value" and ca:
            corrupted = dict(ca)
            change_key = _random.choice(list(corrupted.keys()))
            corrupted[change_key] = _pick_wrong_value(change_key, corrupted[change_key])
            return {"function": cf, "arguments": corrupted}
        else:
            return {"function": cf, "arguments": {}}

    if failure_kind == "skip_call" and len(gold_calls) > 1:
        drop_idx = _random.randint(0, len(gold_calls) - 1)
        corrupted_calls = [c for i, c in enumerate(gold_calls) if i != drop_idx]
    else:
        corrupted_calls = [_corrupt_single_call(c) for c in gold_calls]

    call_blocks = []
    for call in corrupted_calls:
        call_json = json.dumps(call, indent=2, ensure_ascii=False)
        call_blocks.append(f"<tool_call>\n{call_json}\n</tool_call>")
    calls_str = "\n".join(call_blocks)
    return f"<reasoning>\n{reasoning}\n</reasoning>\n{calls_str}"
```

```python
# ─────────────────────────────────────────────────────────────────────────────
# RC-GRPO ↔ TRL integration
#
# BUG FIXED: `RCGRPOTrainer` was referenced (register_reward_tokens(), and
# instantiated as a trainer) but NEVER DEFINED anywhere in the notebook —
# same for GVPOTrainer / AutoToolTrainer / AWPOTrainer / rc_grpo_reward_func
# / rc_grpo_format_func / inject_diverse_reward_tokens. Running the final
# cell would raise NameError immediately.
#
# This fix implements a real, minimal RC-GRPO on top of TRL's GRPOTrainer
# (the class the rest of the notebook is already built around via Unsloth +
# vLLM), rather than the disconnected, never-called, single-prompt manual
# loop (`run_rc_grpo_training`) defined earlier. That manual loop is kept
# above for reference/teaching value but is NOT part of the runnable path.
#
# Design (Algorithm 1 / Eq. 3-9), adapted to TRL's GRPOTrainer:
#   - TRL's GRPOTrainer already implements the GRPO core: group sampling
#     (num_generations=G), group-relative advantage normalization (we
#     align its eps to RC-GRPO's eps_stab), clipped surrogate loss, and the
#     KL penalty against a frozen reference model. We do NOT reimplement
#     these — we reuse TRL's battle-tested, vLLM-accelerated versions.
#   - What TRL does NOT do natively is reward-CONDITIONED generation: i.e.
#     sampling a distinct reward token r_j ~ P_sample(r) per rollout within
#     a group, and conditioning that rollout's generation on r_j (Eq. 3-4).
#   - We achieve this by overriding `_generate_and_score_completions`'s
#     prompt-construction step: for each of the G prompt repeats TRL builds
#     internally, we inject a sampled reward token into the FIRST user
#     turn before generation, matching exactly how RCTP-FT trained the
#     token associations (Appendix B).
#   - The token itself carries no information into the reward function:
#     R(tau) (Eq. 5) only checks state/action correctness, never the
#     conditioning token, so it is stripped before being passed to the
#     reward function via `format_reward`/`composite_reward`.
# ─────────────────────────────────────────────────────────────────────────────

from trl import GRPOTrainer, GRPOConfig
import numpy as _np
import random as _random_module


def sample_reward_tokens_for_group(
    group_size: int,
    high_reward_probability: float = 0.5,
) -> List[str]:
    """Sample G reward tokens from P_sample(r) (Eq. 3)."""
    return [
        HIGH_REWARD_TOKEN if _random_module.random() < high_reward_probability else LOW_REWARD_TOKEN
        for _ in range(group_size)
    ]


def inject_reward_token_into_messages(
    prompt_messages: List[Dict],
    reward_token: str,
) -> List[Dict]:
    """
    Inject `[Reward Goal: <token>]` into the system message only
    (Appendix B), matching RCTP-FT training exactly.
    """
    out = []
    injected = False
    for msg in prompt_messages:
        if msg.get("role") == "system" and not injected:
            out.append({**msg, "content": f"{msg['content']}\n[Reward Goal: {reward_token}]"})
            injected = True
        else:
            out.append(msg)
    return out


class RCGRPOTrainer(GRPOTrainer):
    """
    TRL GRPOTrainer subclass implementing reward-conditioned rollout
    sampling (Eq. 3-4) on top of TRL's existing GRPO core.

    `high_reward_probability` (p in Eq. 3) should be set to the empirical
    success rate of the RCTP-FT dataset (Appendix B.1.1: "we set p = 0.5"
    because their RCTP-FT set was an exact 1:1 success:failure ratio).
    Use `build_rctp_dataset_from_jsonl(...)` and compute
    n_success / (n_success + n_failure) to get the matching value for your
    data instead of hardcoding 0.5 blindly.
    """

    def __init__(self, *args, high_reward_probability: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_reward_probability = high_reward_probability
        # diagnostics, mirrors paper's "advantage spread" tracking (Table 5)
        self._last_reward_tokens: List[str] = []

    @staticmethod
    def register_reward_tokens(tokenizer) -> int:
        """Register <|high_reward|> / <|low_reward|> as additional special
        tokens. Returns the number of NEW tokens added (0 if already
        present), so the caller knows whether resize_token_embeddings is
        needed."""
        existing = set(tokenizer.all_special_tokens)
        to_add = [t for t in REWARD_TOKENS if t not in existing]
        if not to_add:
            return 0
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        return num_added

    def _generate_and_score_completions(self, inputs):
        """
        Override TRL's per-group prompt construction to inject a freshly
        sampled reward token (Eq. 3) into EACH of the G repeated prompts
        before generation, then delegate to the parent implementation for
        actual vLLM generation + reward scoring + advantage computation.

        TRL repeats each unique prompt `num_generations` (= G) times inside
        a batch before calling this method, so `inputs` already contains G
        consecutive copies of the same example. We detect those repeats by
        grouping on a per-example key (here: the 'query' field, which is
        unique per training sample) and inject a distinct token per slot.
        """
        G = self.args.num_generations

        # Group inputs into consecutive chunks of size G (TRL's repeat order)
        conditioned_inputs = []
        for group_start in range(0, len(inputs), G):
            group = inputs[group_start: group_start + G]
            reward_tokens = sample_reward_tokens_for_group(
                len(group), self.high_reward_probability
            )
            self._last_reward_tokens.extend(reward_tokens)
            for example, r_j in zip(group, reward_tokens):
                conditioned_example = dict(example)
                prompt_str = conditioned_example["prompt"]
                conditioned_example["prompt"] = inject_reward_token_into_prompt(
                    prompt_str, r_j
                )
                # carry the token through so reward funcs can log it if useful
                conditioned_example["_reward_token"] = r_j
                conditioned_inputs.append(conditioned_example)

        return super()._generate_and_score_completions(conditioned_inputs)


def rc_grpo_reward_func(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """
    R(tau) = R_state * R_action (Eq. 5), binary in {0, 1}.

    FIXED: ground_truth now arrives as a JSON string column (see
    format_sample_for_grpo) — must json.loads() it first via _parse_gt.
    FIXED: handles abstention — returns 1.0 when gold calls is empty and
    model produces no tool call.
    FIXED: null-function calls in ground_truth are normalized to empty
    calls list via _parse_gt -> normalize_ground_truth.

    NOTE: this intentionally does NOT look at the reward-conditioning token
    — per the paper, the trajectory reward is a property of the
    state/action outcome only, never the token that was used to *steer*
    generation (Sec. 3.3, "Trajectory-Level Reward Function").
    """
    rewards = []
    for completion, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        gold_calls = [
            {"function": c["function"], "arguments": c.get("arguments", {})}
            for c in gt.get("calls", [])
        ]
        if not gold_calls:
            agent_calls = extract_all_calls(completion)
            rewards.append(1.0 if not agent_calls else 0.0)
            continue

        agent_calls = extract_all_calls(completion)

        if not agent_calls:
            rewards.append(0.0)
            continue

        r_action = compute_action_coverage_reward(agent_calls, gold_calls)
        rewards.append(float(r_action))

    return rewards

def rc_grpo_format_func(completions: list[str], **kwargs) -> list[float]:
    """Lightweight format shaping reward (kept separate from the binary
    trajectory reward so R(tau) used for reward-token quantization, Eq. 1,
    stays exactly binary)."""
    return format_reward(completions, **kwargs)
```

```python
# ===================== base_trainer.py =====================
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

try:
    import jsonlines
except ImportError:
    jsonlines = None

# ─────────────────────────────────────────────────────────────────────
# Chat template — supports system, user, assistant, tool, retriever
# ─────────────────────────────────────────────────────────────────────

CUSTOM_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{{- '<|im_start|>system\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- elif message.role == 'user' %}"
    "{{- '<|im_start|>user\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>tool\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- elif message.role == 'retriever' %}"
    "{{- '<|im_start|>retriever\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\n' + message.content + '\n<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- endif %}"
)


def patch_tokenizer_for_custom_roles(tokenizer) -> None:
    """Register custom chat template and reward special tokens."""
    tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE

    # --- FIX: always register reward tokens ---
    existing_special = tokenizer.additional_special_tokens or []
    tokens_to_add = [t for t in REWARD_TOKENS if t not in existing_special]
    if tokens_to_add:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_special + tokens_to_add}
        )
        print(f"[base_trainer] Added reward tokens: {tokens_to_add}")

    print("[base_trainer] Custom chat template registered (supports 'retriever' role).")


# ─────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — updated to support both single and sequential workflows
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a telecom network operations assistant. \
You will receive a user query, a list of available functions with their \
parameter schemas, and relevant argument values retrieved from a knowledge base.

Your task:
1. Read the user query carefully
2. Review the available functions and retrieved argument values in the \
<|im_start|>retriever<|im_end|> block
3. Reason step by step about which function(s) to call and what argument \
values to use
4. Output your reasoning and the function call(s)

REQUIRED OUTPUT FORMAT — use these exact tags:
<reasoning>
Step-by-step analysis:
- What the user is asking for
- Which function(s) best match and why
- Which argument values from the retriever match the query
- Final argument assignments
</reasoning>
<tool_call>
{"function": "<function_name>", "arguments": {"<param1>": "<value1>", ...}}
</tool_call>

If the query requires MULTIPLE sequential calls, output multiple <tool_call> blocks:
<tool_call>
{"function": "<function_name_1>", "arguments": {...}}
</tool_call>
<tool_call>
{"function": "<function_name_2>", "arguments": {...}}
</tool_call>

STRICT RULES:
- Call ONLY functions from the retriever list
- Include ALL required parameters
- Use argument values from the retriever when they match the query
- Do NOT invent functions or parameters not in the retriever block
- If the query is unanswerable with available functions, output:
<reasoning>Explanation of why no function is appropriate.</reasoning>
<tool_call>null</tool_call>
"""


# ─────────────────────────────────────────────────────────────────────
# Prompt building helpers
# ─────────────────────────────────────────────────────────────────────

def build_function_description(func_name: str, schema: dict) -> str:
    lines = [f"### {func_name}"]
    lines.append(
        f"Description: {schema.get('description', 'No description available')}"
    )
    params = schema.get("parameters", {})
    constraints = schema.get("constraints", {})
    if params:
        lines.append("Parameters:")
        for pname, pinfo in params.items():
            if isinstance(pinfo, dict):
                ptype = pinfo.get("type", "any")
                required = "(required)" if pinfo.get("required") else "(optional)"
                desc = pinfo.get("description", "")
                line = f"  - {pname} [{ptype}] {required}: {desc}"
                # --- FIX: surface enum constraints directly ---
                param_con = constraints.get(pname, {})
                if "enum" in param_con:
                    line += f"  Allowed values: {param_con['enum']}"
                lines.append(line)
            else:
                lines.append(f"  - {pname}: {pinfo}")
    if constraints:
        # show non-enum constraints
        other_con = {
            k: v for k, v in constraints.items()
            if not (isinstance(v, dict) and list(v.keys()) == ["enum"])
        }
        if other_con:
            lines.append(
                f"Constraints: {json.dumps(other_con, ensure_ascii=False)}"
            )
    return "\n".join(lines)


def build_argument_values_block(argument_values: dict[str, list]) -> str:
    if not argument_values:
        return ""
    lines = ["## Relevant Argument Values"]
    for param_name, matches in argument_values.items():
        if not matches:
            continue
        lines.append(f"\n### {param_name}")
        for m in matches:
            if hasattr(m, "code"):
                label_str = m.label
                if m.alt_label:
                    label_str += f" / {m.alt_label}"
                lines.append(f"  - {m.code} → {label_str}  [{m.group}]")
            else:
                code = m.get("code", "")
                label = m.get("label", "")
                group = m.get("group", "")
                lines.append(f"  - {code} → {label}  [{group}]")
    return "\n".join(lines)


def build_retriever_block(
    function_names: list[str],
    function_library: dict,
    argument_values: dict[str, list] | None = None,
    include_all_threshold: int = 10,
) -> str:
    lines = ["## Available Functions\n"]
    for fn in function_names:
        if fn in function_library:
            lines.append(build_function_description(fn, function_library[fn]))
            lines.append("")

    if argument_values:
        val_lines = ["## Relevant Argument Values"]
        for param_name, matches in argument_values.items():
            if not matches:
                continue
            val_lines.append(f"\n### {param_name}")
            for m in matches:
                if hasattr(m, "code"):
                    label_str = m.label
                    if m.alt_label:
                        label_str += f" / {m.alt_label}"
                    val_lines.append(
                        f"  - {m.code} → {label_str}  [{m.group}]"
                    )
                else:
                    code = m.get("code", "")
                    label = m.get("label", "")
                    group = m.get("group", "")
                    val_lines.append(f"  - {code} → {label}  [{group}]")
        if len(val_lines) > 1:
            lines.append("\n".join(val_lines))

    return "\n".join(lines)


def build_messages_for_grpo(
    query: str,
    function_names: list[str],
    function_library: dict,
    argument_values: dict[str, list] | None = None,
    include_all_threshold: int = 10,
) -> list[dict]:
    retriever_content = build_retriever_block(
        function_names,
        function_library,
        argument_values,
        include_all_threshold,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "retriever", "content": retriever_content},
    ]


def load_model(
    base_model_name: str = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    fast_inference: bool = False,
    adapter_model_path: str | None = None,
    mode: str = "train",
    lora_rank: int = 16,
    lora_target_modules: list[str] | None = None,
    lora_dropout: float = 0.0,
    env_name: str = "local",
) -> tuple:
    """
    Unified model loader: load a base model with optional LoRA adapter.

    Three modes:
      1. adapter_model_path=... + mode="train" → checkpoint resume for continued training
      2. adapter_model_path=... + mode="inference" → checkpoint for evaluation
      3. adapter_model_path=None + mode="inference" → base model only (benchmarking)
      4. adapter_model_path=None + mode="train" → fresh LoRA from scratch (SFT/RCTP-FT)

    Loading order (critically preserves embedding/adapter compatibility):
      1. Determine tokenizer source (checkpoint vs base model)
      2. Load base model via FastLanguageModel.from_pretrained (fast_inference handles internal patching)
      3. Resize embeddings if adapter_model_path is given (checkpoint resume)
      4. Load existing adapter or create a fresh LoRA via get_peft_model
      5. Enable inference mode if mode == "inference"

    Args:
        base_model_name: Name or path of the original base model.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Whether to quantize to 4-bit.
        fast_inference: Enable vLLM fast inference (required for GRPO).
        adapter_model_path: Path to a trained adapter checkpoint. If None,
            creates a fresh LoRA via get_peft_model (from-scratch training).
        mode: "train" (keep trainable) or "inference" (for_inference).
        lora_rank: LoRA rank (used only when adapter_model_path is None).
        lora_target_modules: LoRA target modules (fresh LoRA only).
        lora_dropout: LoRA dropout (fresh LoRA only).
        env_name: Environment name for Kaggle path override.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    # ── 0. Kaggle path override ───────────────────────────────────────
    if env_name == "kaggle" and base_model_name == "unsloth/Qwen3-4B-Instruct-2507":
        base_model_name = "/kaggle/input/models/dzung271828/unsloth/transformers/default/3/qwen3-4b-instruct-2507"

    # ── 1. Load tokenizer from checkpoint or base model ───────────────
    if adapter_model_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # ── 2. Load the base model (fast_inference handles internal GRPO patching) ─
    kwargs = dict(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        dtype=None,
        gpu_memory_utilization=0.3 if os.environ.get("UNSLOTH_VLLM_STANDBY", "0") != "1" else 0.8,
    )
    if adapter_model_path is None and not fast_inference and mode == "train":
        kwargs["max_lora_rank"] = lora_rank

    model, _ = FastLanguageModel.from_pretrained(**kwargs)

    # ── 4. Load adapter, base model only, or create fresh LoRA ────────
    if adapter_model_path is not None:
        # Checkpoint resume: resize embeddings to match checkpoint tokenizer
        model.resize_token_embeddings(len(tokenizer))
        model.load_adapter(adapter_model_path, adapter_name="default")
    elif mode == "inference":
        # Base model only (no adapter) — for benchmarking the base model
        pass
    else:
        # Fresh training: create LoRA from scratch
        target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank * 2,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        # Register custom chat template + reward tokens for fresh training
        patch_tokenizer_for_custom_roles(tokenizer)
        model.resize_token_embeddings(len(tokenizer))
        print(
            f"[base_trainer] Tokenizer vocab size: {len(tokenizer)} "
            f"(includes reward tokens: {REWARD_TOKENS})"
        )

    # ── 5. Enable inference mode if needed ────────────────────────────
    if mode == "inference":
        FastLanguageModel.for_inference(model)

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
# GRPO config builder — SINGLE DEFINITION (BUG FIXED: same shadowing
# problem. The surviving (second) definition used epsilon_high=0.28,
# i.e. ASYMMETRIC clipping (a DAPO-style trick), which the RC-GRPO paper
# does NOT use. Table 10 specifies a single symmetric Clip Ratio = 0.2
# for both models. The first definition correctly omitted epsilon_high,
# but was shadowed and dead code.
# ─────────────────────────────────────────────────────────────────────

from trl import GRPOConfig
from vllm import SamplingParams


def build_grpo_config(config: dict, output_dir: str | None = None) -> GRPOConfig:
    train_cfg = config.get("training", {})
    grpo_cfg = config.get("grpo", {})
    data_cfg = config.get("data", {})

    vllm_params = SamplingParams(
        temperature=grpo_cfg.get("temperature", 1.0),
        top_p=0.95,
        min_p=0.05,
        seed=train_cfg.get("seed", 3407),
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )

    return GRPOConfig(
        output_dir=output_dir or train_cfg.get("output_dir", "outputs/model"),
        learning_rate=train_cfg.get("learning_rate", 1e-6),  # Table 10: 1e-6
        adam_beta1=train_cfg.get("adam_beta1", 0.9),
        adam_beta2=train_cfg.get("adam_beta2", 0.99),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim=train_cfg.get("optim", "adamw_8bit"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        num_generations=train_cfg.get("num_generations", 5),  # Table 10: G=5
        max_prompt_length=data_cfg.get("max_prompt_length", 1792),
        max_completion_length=data_cfg.get("max_completion_length", 256),
        temperature=grpo_cfg.get("temperature", 1.0),
        # FIXED: symmetric clipping only (Table 10: Clip Ratio = 0.2).
        # epsilon_high removed entirely — do NOT pass it, TRL will default
        # to symmetric clipping when only `epsilon` is set.
        epsilon=grpo_cfg.get("epsilon", 0.2),
        beta=grpo_cfg.get("kl_coefficient", 0.1),  # Table 10: KL coeff beta=0.1
        loss_type=grpo_cfg.get("loss_type", "grpo"),
        mask_truncated_completions=grpo_cfg.get("mask_truncated_completions", True),
        vllm_sampling_params=vllm_params,
        # ── vLLM KV cache reduction ────────────────────────────────────
        # Cap KV cache to exactly prompt+completion length rather than
        # full max_seq_length (8192), saving VRAM for training activations.
        vllm_max_model_length=data_cfg.get("max_prompt_length", 3584) + data_cfg.get("max_completion_length", 256),
        # Tight vLLM memory fraction — training model needs more now with batch=2, seq=4096.
        vllm_gpu_memory_utilization=0.5,
        # Offload vLLM KV cache to CPU during optimizer steps (frees VRAM).
        vllm_enable_sleep_mode=True,
        max_steps=train_cfg.get("max_steps", 500),
        save_steps=train_cfg.get("save_steps", 100),
        logging_steps=train_cfg.get("logging_steps", 1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 3407),
        bf16=torch.cuda.is_bf16_supported(),
    )


# ─────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────

def build_trl_reward_functions(algorithm: str = "rc_grpo"):
    """
    Returns the list of reward functions compatible with TRL's GRPOTrainer
    for the selected algorithm.

    Each function signature: fn(completions: list[str], **kwargs) -> list[float]
    kwargs includes 'ground_truth' passed through from the dataset columns.
    """
    if algorithm == "rc_grpo":
        return [rc_grpo_reward_func, rc_grpo_format_func]
    # default / vanilla GRPO
    return [function_reward, argument_reward, format_reward]

# ─────────────────────────────────────────────────────────────────────
# Reward-token injection helper for TRL GRPOTrainer subclass
# ─────────────────────────────────────────────────────────────────────

def inject_reward_token_into_prompt(
    prompt: str,
    reward_token: str,
) -> str:
    """
    Inject [Reward Goal: <token>] right after the system message content
    (before the system's <|im_end|> marker) in a pre-built prompt string.

    For use when subclassing TRL's GRPOTrainer to add RC-GRPO support.
    """
    # Find the system message end marker and inject right after system content
    system_end_marker = "<|im_start|>system\n"
    system_close_marker = "\n<|im_end|>\n"
    sys_start = prompt.find(system_end_marker)
    sys_close = prompt.find(system_close_marker, sys_start + len(system_end_marker)) if sys_start != -1 else -1
    if sys_start != -1 and sys_close != -1:
        # Inject right after the system content ends (before close marker)
        inject_str = f"\n[Reward Goal: {reward_token}]"
        return prompt[:sys_close] + inject_str + prompt[sys_close:]
    # fallback: append to end
    return prompt + f"\n[Reward Goal: {reward_token}]\n"


```

```python
# ===================== metrics.py =====================
def compute_all_metrics(response: str, ground_truth: dict, sandbox, latency_ms: float,
                        cost_estimate: float, function_library: dict) -> dict[str, float]:
    gt = normalize_ground_truth(ground_truth)
    workflow = gt.get("workflow", "single_call")
    gold_calls = [
        {"function": c["function"], "arguments": c.get("arguments", {})}
        for c in gt.get("calls", [])
    ]
    agent_calls = extract_all_calls(response)
    schema_val = 1.0 if agent_calls else 0.0

    if gold_calls:
        # Per-call metrics: best match for each gold call
        func_hits = 0
        arg_scores = []
        for gc in gold_calls:
            best_func = 0.0
            best_arg = 0.0
            for ac in agent_calls:
                if ac.get("function") == gc["function"]:
                    best_func = 1.0
                    best_arg = max(best_arg, args_accuracy(ac, gc.get("arguments", {})))
            func_hits += best_func
            arg_scores.append(best_arg)
        func_sel_acc = func_hits / len(gold_calls)
        arg_acc = sum(arg_scores) / len(arg_scores)

        task_success = float(compute_action_coverage_reward(agent_calls, gold_calls))

        exec_success = 0.0
        if sandbox is not None:
            exec_results = sandbox.execute_all(response)
            exec_success = float(all(exec_results)) if exec_results else 0.0
        else:
            exec_success = float(len(agent_calls) > 0)
    else:
        # Abstention — expect no tool call
        func_sel_acc = float(len(agent_calls) == 0)
        arg_acc = float(len(agent_calls) == 0)
        task_success = float(len(agent_calls) == 0)
        exec_success = float(len(agent_calls) == 0)

    hallucinated = 0.0
    for ac in agent_calls:
        called_fn = ac.get("function", "")
        if called_fn and called_fn not in function_library:
            hallucinated = 1.0
            break

    abstention_acc = float("nan")
    if workflow == "abstention" or not gold_calls:
        produced_call = extract_call(response)
        abstention_acc = 1.0 if (produced_call is None or produced_call == "null") else 0.0

    return {
        "function_selection_accuracy": func_sel_acc,
        "argument_accuracy": arg_acc,
        "schema_validity": schema_val,
        "execution_success_rate": exec_success,
        "task_success_rate": task_success,
        "hallucinated_call_rate": hallucinated,
        "abstention_accuracy": abstention_acc,
        "latency_ms": latency_ms,
        "cost_per_query_usd": cost_estimate,
    }
def aggregate_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results if isinstance(r[k], (int, float)) and not np.isnan(r[k])]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[f"{k}__std"] = float(np.std(vals))
            agg[f"{k}__count"] = len(vals)
        else:
            agg[k] = float("nan")
    return agg

def estimate_cost(prompt: str, response: str, price_per_1k_tokens: float = 0.0002) -> float:
    total_chars = len(prompt) + len(response)
    tokens_est = total_chars / 1.3
    return (tokens_est / 1000) * price_per_1k_tokens

def evaluate_model_vllm(
    model_path: str,
    test_dataset_path: str,
    function_library: dict,
    retriever: FunctionRetriever,
    sandbox: Sandbox,
    top_k: int = 5,
    max_new_tokens: int = 512,
    model_name_tag: str = "model",
    use_dataset_retrieval: bool = True,
    argument_values: dict | None = None,
    condition_on_high_reward: bool = True,
    batch_size: int = 32,
) -> dict:
    """
    Batched evaluation using vLLM for ~10x speedup over serial HF generate.

    Loads model with fast_inference=True, uses vLLM's LLM.generate() with
    continuous batching. Returns same metrics dict as evaluate_model().
    """
    logger = get_logger(__name__)
    logger.info(f"[vLLM-Bench] Loading model from {model_path}")

    from vllm import LLM, SamplingParams

    # Load with fast_inference=True for vLLM backend
    model, tokenizer = load_model(
        adapter_model_path=model_path,
        mode="inference",
        fast_inference=True,
        env_name=ENV_NAME,
    )

    val_retriever = None
    if argument_values is not None:
        val_retriever = ArgumentValueRetriever(argument_values)

    test_samples = []
    with jsonlines.open(test_dataset_path) as reader:
        for obj in reader:
            test_samples.append(obj)

    logger.info(f"[vLLM-Bench] {model_name_tag}: evaluating {len(test_samples)} samples (batch_size={batch_size})")

    # Build all prompts first
    prompts = []
    sample_metas = []
    for sample in test_samples:
        query = sample["query"]
        gt = sample.get("ground_truth", {})
        if use_dataset_retrieval and sample.get("retrieved_functions"):
            retrieved = sample["retrieved_functions"]
        else:
            retrieved = retriever.retrieve(query, k=top_k)
        if val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(query, retrieved, function_library)
        else:
            arg_vals = sample.get("retrieved_argument_values")

        messages = build_messages_for_grpo(query, retrieved, function_library, arg_vals)

        if condition_on_high_reward and HIGH_REWARD_TOKEN in tokenizer.all_special_tokens:
            messages = inject_reward_token_into_messages(messages, HIGH_REWARD_TOKEN)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        sample_metas.append({"sample": sample, "gt": gt})

    # Batch generation with vLLM
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )

    # Create vLLM LLM from the same model
    # Use the underlying model for vLLM inference
    t_start = time.perf_counter()
    outputs = model.fast_generate(prompts, sampling_params)
    gen_time = time.perf_counter() - t_start
    logger.info(f"[vLLM-Bench] Batch generation completed in {gen_time:.1f}s ({gen_time/len(prompts):.3f}s/sample)")

    # Process results
    results = []
    for idx, (output, meta) in enumerate(zip(outputs, sample_metas)):
        sample = meta["sample"]
        gt = meta["gt"]
        response = output.outputs[0].text
        t0 = time.perf_counter()
        # latency is approximated as total_time / batch
        latency = (gen_time / len(prompts)) * 1000.0
        cost = estimate_cost(prompts[idx], response)
        metrics = compute_all_metrics(response, gt, sandbox, latency, cost, function_library)
        metrics["sample_id"] = sample.get("id", "")
        results.append(metrics)

    agg = aggregate_metrics(results)
    logger.info(f"[vLLM-Bench] {model_name_tag} aggregate: {agg}")
    return {"model": model_name_tag, "per_sample": results, "aggregate": agg}


# ===================== benchmark.py =====================
# ===================== benchmark.py (FIXED) =====================
# BUG FIXED: paper Sec. 4.1 — "At inference, we condition on r =
# <|high_reward|> for optimal performance." The original evaluate_model
# built plain prompts with build_messages_for_grpo() and never injected
# the reward-goal token, so an RC-GRPO-trained model would be evaluated
# completely unconditioned (off-distribution relative to how it was
# trained to behave optimally).

def evaluate_model(
    model_path: str,
    test_dataset_path: str,
    function_library: dict,
    retriever: FunctionRetriever,
    sandbox: Sandbox,
    top_k: int = 5,
    max_new_tokens: int = 512,
    model_name_tag: str = "model",
    use_dataset_retrieval: bool = True,
    argument_values: dict | None = None,
    condition_on_high_reward: bool = True,
) -> dict:
    logger = get_logger(__name__)
    logger.info(f"[Benchmark] Loading model from {model_path}")
    model, tokenizer = load_model(
        adapter_model_path=model_path,
        mode="inference",
        fast_inference=True,
        env_name=ENV_NAME,
    )

    val_retriever = None
    if argument_values is not None:
        val_retriever = ArgumentValueRetriever(argument_values)

    test_samples = []
    with jsonlines.open(test_dataset_path) as reader:
        for obj in reader:
            test_samples.append(obj)

    logger.info(f"[Benchmark] {model_name_tag}: evaluating {len(test_samples)} samples")
    results = []
    for sample in tqdm(test_samples, desc=f"Eval [{model_name_tag}]"):
        query = sample["query"]
        gt = sample.get("ground_truth", {})
        if use_dataset_retrieval and sample.get("retrieved_functions"):
            retrieved = sample["retrieved_functions"]
        else:
            retrieved = retriever.retrieve(query, k=top_k)
        if val_retriever is not None:
            arg_vals = val_retriever.retrieve_for_functions(query, retrieved, function_library)
        else:
            arg_vals = sample.get("retrieved_argument_values")

        messages = build_messages_for_grpo(query, retrieved, function_library, arg_vals)

        # FIXED: condition on <|high_reward|> at inference (Sec. 4.1), only
        # meaningful/safe if the reward tokens are actually registered
        # (i.e. this checkpoint went through RCTP-FT / RC-GRPO).
        if condition_on_high_reward and HIGH_REWARD_TOKEN in tokenizer.all_special_tokens:
            messages = inject_reward_token_into_messages(messages, HIGH_REWARD_TOKEN)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        response = generate_response(model, tokenizer, prompt, max_new_tokens, temperature=0.0, do_sample=False)
        latency = (time.perf_counter() - t0) * 1000.0
        cost = estimate_cost(prompt, response)
        metrics = compute_all_metrics(response, gt, sandbox, latency, cost, function_library)
        metrics["sample_id"] = sample.get("id", "")
        results.append(metrics)

    agg = aggregate_metrics(results)
    logger.info(f"[Benchmark] {model_name_tag} aggregate: {agg}")
    return {"model": model_name_tag, "per_sample": results, "aggregate": agg}

```

```python
# ===================== report_generator.py =====================
METRIC_DISPLAY_NAMES = {
    "function_selection_accuracy": "Func. Selection Acc.",
    "argument_accuracy": "Arg. Accuracy",
    "schema_validity": "Schema Validity",
    "execution_success_rate": "Exec. Success Rate",
    "task_success_rate": "Task Success Rate",
    "hallucinated_call_rate": "Hallucination Rate ↓",
    "abstention_accuracy": "Abstention Acc.",
    "latency_ms": "Latency (ms) ↓",
    "cost_per_query_usd": "Cost/Query (USD) ↓",
}

def generate_report(eval_results: list[dict], output_dir: str = "outputs/evaluation_reports") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in eval_results:
        model = result["model"]
        agg = result["aggregate"]
        row = {"Model": model}
        for k, display in METRIC_DISPLAY_NAMES.items():
            row[display] = round(agg.get(k, float("nan")), 4)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    print("\n" + "=" * 80)
    print("  EVALUATION REPORT — Telecom Tool-Calling with RL")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))
    csv_path = out / "metrics_summary.csv"
    df.to_csv(csv_path)
    print(f"[Report] CSV saved → {csv_path}")
    json_path = out / "full_results.json"
    with open(json_path, "w") as fh:
        json.dump(eval_results, fh, indent=2, default=str)
    print(f"[Report] JSON saved → {json_path}")
    # Optional plots
    try:
        _plot_bar_comparison(df, out)
        _plot_radar(df, out)
    except Exception as e:
        print(f"Plot generation failed: {e}")

def _plot_bar_comparison(df: pd.DataFrame, out: Path) -> None:
    core_metrics = ["Func. Selection Acc.", "Arg. Accuracy", "Schema Validity",
                    "Exec. Success Rate", "Task Success Rate"]
    plot_df = df[[c for c in core_metrics if c in df.columns]]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.T.plot(kind="bar", ax=ax, width=0.7, colormap="tab10")
    ax.set_title("Model Comparison – Core Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower right")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(out / "bar_comparison.png", dpi=150)
    plt.close(fig)

def _plot_radar(df: pd.DataFrame, out: Path) -> None:
    radar_metrics = ["Func. Selection Acc.", "Arg. Accuracy", "Schema Validity",
                     "Exec. Success Rate", "Task Success Rate", "Abstention Acc."]
    categories = [c for c in radar_metrics if c in df.columns]
    N = len(categories)
    if N < 3:
        return
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"polar": True})
    cmap = plt.get_cmap("tab10")
    for i, (model, row) in enumerate(df.iterrows()):
        vals = [row.get(c, 0.0) for c in categories]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=cmap(i), label=model)
        ax.fill(angles, vals, alpha=0.1, color=cmap(i))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Algorithm Radar Comparison", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(out / "radar_comparison.png", dpi=150)
    plt.close(fig)
```

```python
import os
if ENV_NAME == "colab":
    # Create the data directory
    os.makedirs('/content/data', exist_ok=True)
    # The user will upload the generated.zip file to the /content directory in Colab
    # Unzip the file into the data folder
    !unzip -o /content/generated.zip -d /content/ToolFormer/data
```

```python
# ===================== CONFIGURATION =====================
MODE = "sft"  # one of: "sft", "grpo", "rc_grpo", "rctp_ft"
assert MODE in ("sft", "grpo", "rc_grpo", "rctp_ft"), \
    f"Unknown MODE: {MODE}. Choose from: sft, grpo, rc_grpo, rctp_ft"
EVAL_USE_VLLM = True  # Use vLLM batch eval for ~10x speedup

DATA_DIR = Path(DATA_MOUNT)
# Kaggle: /kaggle/input/ is read-only — skip mkdir, data is pre-mounted
if ENV_NAME != "kaggle":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

FUNCTION_LIBRARY_PATH = DATA_DIR / "function_library.json"
ARGUMENT_VALUES_PATH = DATA_DIR / "argument_values.json"  # optional


TRAIN_CONFIG = {
    "model": {
        "name": "unsloth/Qwen3-4B-Instruct-2507",
        "max_seq_length": 8192,
        "load_in_4bit": True,
        "fast_inference": False,
        "gpu_memory_utilization": 0.8,
        "full_finetuning": False,
    },
    "lora": {
        "r": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
    },
    "training": {
        "output_dir": f"outputs/{MODE}_model",
        # Table 10 (RC-GRPO RL stage hyperparameters)
        "learning_rate": 5e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.001,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_generations": 5,         # Table 10: Group Size G = 5
        "max_steps": 10,             # reduce for quick test; increase for full run
        "save_steps": 50,
        "logging_steps": 1,
        "max_grad_norm": 1.0,
        "report_to": "none",
        "seed": 3407,
    },
    "data": {
        "train_path": str(DATA_DIR / "train_dataset_cleaned.jsonl"),
        "test_path": str(DATA_DIR / "test_dataset_cleaned.jsonl"),
        "sft_path": str(DATA_DIR / "sft_dataset.jsonl"),
        "grpo_path": str(DATA_DIR / "grpo_dataset.jsonl"),
        "rctp_path": str(DATA_DIR / "rctp_dataset.jsonl"),
        "max_prompt_length": 7680,
        "max_completion_length": 512,
        "include_all_threshold": 5,
        "eval_batch_size": 32,
    },
    "grpo": {
        "temperature": 1.0,
        "epsilon": 0.2,            # Table 10: symmetric Clip Ratio = 0.2
        "kl_coefficient": 0.1,     # Table 10: KL Coefficient beta = 0.1
        "loss_type": "grpo",
        "mask_truncated_completions": True,
    },
    "sft": {
        "output_dir": "outputs/sft_model",
        "learning_rate": 2e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "max_seq_length": 8192,
    },
    # RCTP-FT (Stage 1) hyperparameters — Table 11
    "rctp_ft": {
        "output_dir": "outputs/rctp_ft_model",
        "learning_rate": 2e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_epochs": 3,
        "failures_per_expert": 1,  # -> exact 1:1 success:failure ratio (Table 6)
    },
}
# Load function library and argument values (if available)
function_library = load_function_library(FUNCTION_LIBRARY_PATH)
print(f"Loaded {len(function_library)} functions")

# Load argument values catalog as ValueCatalog objects (needed by retrievers)
argument_values_catalog = None
if ARGUMENT_VALUES_PATH.exists():
    argument_values_catalog = load_catalog_from_json(str(ARGUMENT_VALUES_PATH))
    print(f"Loaded argument values catalog with {len(argument_values_catalog)} parameters")
else:
    print("No argument_values.json; argument value retrieval will be skipped.")

# ===================== Load model & register reward tokens =====================
# Resolve output directory from active MODE
MODE_OUTPUT_DIR = TRAIN_CONFIG["training"]["output_dir"]
model, tokenizer = load_model(
    base_model_name=TRAIN_CONFIG["model"]["name"],
    max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],
    load_in_4bit=TRAIN_CONFIG["model"].get("load_in_4bit", True),
    fast_inference=False,
    adapter_model_path=None,
    mode="train",
    lora_rank=TRAIN_CONFIG["lora"]["r"],
    lora_target_modules=TRAIN_CONFIG["lora"]["target_modules"],
    lora_dropout=TRAIN_CONFIG["lora"].get("lora_dropout", 0.0),
    env_name=ENV_NAME,
)
print("Model and tokenizer loaded.")

print("Tokenizer special tokens:")
print(f"  bos_token: {tokenizer.bos_token}")
print(f"  eos_token: {tokenizer.eos_token}")
print(f"  pad_token: {tokenizer.pad_token}")
print(f"  additional_special_tokens: {tokenizer.additional_special_tokens}")



```

```python
# ── Smart truncation safety net ─────────────────────────────────────
def smart_truncate(text, tokenizer, max_seq_length, verbose=True):
    """Truncate pre-assistant content to preserve the full assistant response."""
    assistant_marker = "<|im_start|>assistant\n"
    idx = text.find(assistant_marker)
    if idx == -1:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_seq_length)
        return tokenizer.decode(tokens, skip_special_tokens=False)
    before = text[:idx]
    after = text[idx:]
    full_len = len(tokenizer.encode(text))
    if full_len <= max_seq_length:
        return text
    before_ids = tokenizer.encode(before, add_special_tokens=False)
    after_ids = tokenizer.encode(after, add_special_tokens=False)
    if len(after_ids) >= max_seq_length - 1:
        if verbose:
            print(f"  [smart_truncate] WARNING: assistant response too long, fallback to right-truncation")
        tokens = tokenizer.encode(text, truncation=True, max_length=max_seq_length)
        return tokenizer.decode(tokens, skip_special_tokens=False)
    max_before = max_seq_length - 1 - len(after_ids)
    truncated_before = tokenizer.decode(before_ids[:max_before], skip_special_tokens=False)
    result = truncated_before + after
    if verbose:
        final_len = len(tokenizer.encode(result))
        print(f"  [smart_truncate] {full_len} -> {final_len} tokens (before: {len(before_ids)}->{max_before})")
    return result

```

```python
# ===================== SFT TRAINING =====================
def load_jsonl(path: str) -> list[dict]:
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

if MODE == "sft":
    print("\n" + "=" * 70)
    print("SFT: Supervised Fine-Tuning on expert demonstrations")
    print("=" * 70)

    sft_cfg = TRAIN_CONFIG["sft"]
    raw_records = load_jsonl(TRAIN_CONFIG["data"]["sft_path"])
    from datasets import Dataset

    # Use full text field directly; train_on_responses_only will mask
    # everything except the assistant response.
    # ── smart_truncate (legacy) ───────────────────────────────────────
    # smart_truncate is disabled because max_seq_length=8192 comfortably
    # fits the full prompt + response with top-5 functions and only the
    # needed argument values (per-sample retrieved_argument_values).
    # Enable if max_seq_length is reduced or the dataset grows.
    # print(f"  [SFT] Applying smart_truncate (max_seq_length={sft_cfg['max_seq_length']})...")
    # before = len(raw_records)
    # raw_records = [{"text": smart_truncate(rec["text"], tokenizer, sft_cfg["max_seq_length"])} for rec in raw_records]
    # ───────────────────────────────────────────────────────────────────
    dataset = Dataset.from_list(raw_records)

    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=sft_cfg["output_dir"],
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
        num_train_epochs=sft_cfg["num_epochs"],
        learning_rate=sft_cfg["learning_rate"],
        max_seq_length=sft_cfg["max_seq_length"],
        max_grad_norm=1.0,
        weight_decay=0.001,
        optim="adamw_8bit",
        seed=3407,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    # Use Unsloth's train_on_responses_only to mask prompt tokens,
    # so the model only learns from the assistant response.
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("=== SFT dataset sample ===")
    for k, v in dataset[0].items():
        print(f"  {k}: {str(v)[:200]}")
    print("===========================")

    trainer.train()
    trainer.save_model(sft_cfg["output_dir"])
    tokenizer.save_pretrained(Path(sft_cfg["output_dir"]))
    print(f"[SFT] Model saved -> {sft_cfg['output_dir']}")
```

```python
!zip -r outputs/sft_model.zip outputs/sft_model
from IPython.display import FileLink
FileLink("outputs/sft_model.zip")
```

```python
# ===================== STAGE 1: RCTP-FT (only for rc_grpo) =====================
SAMPLE_LOG = get_logger("sample")
high_reward_probability = 0.5  # overwritten below from real data stats

if MODE == "rctp_ft":
    print("\n" + "=" * 70)
    print("STAGE 1: Reward-Conditioned Trajectory Policy (RCTP) Fine-tuning")
    print("=" * 70)

    rctp_cfg = TRAIN_CONFIG["rctp_ft"]

    # Inline: Trajectory dataclass + helper (standalone, no scripts/ dependency)
    def binary_reward_to_token(reward: int) -> str:
        return HIGH_REWARD_TOKEN if reward == 1 else LOW_REWARD_TOKEN

    @dataclass
    class Trajectory:
        prompt_messages: list[dict[str, str]]
        response_text: str
        reward: int
        reward_token: str = field(init=False)

        def __post_init__(self):
            self.reward_token = binary_reward_to_token(self.reward)

    rctp_records = load_jsonl(TRAIN_CONFIG["data"]["rctp_path"])
    rctp_trajectories = [Trajectory(**rec) for rec in rctp_records]

    # Eq. 3: p should match the RCTP-FT dataset's empirical success rate
    n_success = sum(1 for t in rctp_trajectories if t.reward == 1)
    n_total = len(rctp_trajectories)
    high_reward_probability = n_success / max(1, n_total)
    print(f"[Stage 1] p (high_reward_probability for Eq. 3) = {high_reward_probability:.4f} "
          f"({n_success}/{n_total} success)")

    # ── Stage 1 via SFTTrainer ──
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    def _format_trajectory(trajectory):
        """
        Build reward-conditioned text: inject [Reward Goal: <token>] into the
        system message (Appendix B), then append the assistant response.
        """
        reward_prefix = f"[Reward Goal: {trajectory.reward_token}]"
        messages = []
        injected = False
        for msg in trajectory.prompt_messages:
            if msg["role"] == "system" and not injected:
                messages.append(
                    {"role": "system", "content": f"{msg['content']}\n{reward_prefix}"}
                )
                injected = True
            else:
                messages.append(msg)
        messages.append({"role": "assistant", "content": trajectory.response_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    formatted_texts = [_format_trajectory(t) for t in rctp_trajectories]
    # ── smart_truncate (legacy) ───────────────────────────────────────
    # Disabled: max_seq_length=8192 fits all samples with the compact
    # per-sample argument values. Enable if max_seq_length is reduced.
    # max_seq_len = TRAIN_CONFIG["model"]["max_seq_length"]
    # print(f"  [RCTP-FT] Applying smart_truncate (max_seq_length={max_seq_len})...")
    # formatted_texts = [smart_truncate(t, tokenizer, max_seq_len) for t in formatted_texts]
    # ────────────────────────────────────────────────────────────────────
    train_dataset = Dataset.from_list([{"text": t} for t in formatted_texts])

    training_args = SFTConfig(
        output_dir=rctp_cfg["output_dir"],
        per_device_train_batch_size=rctp_cfg["batch_size"],
        gradient_accumulation_steps=rctp_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=rctp_cfg["num_epochs"],
        learning_rate=rctp_cfg["learning_rate"],
        max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],  # ← was 1792 (too short for prompt+completion)
        max_grad_norm=1.0,
        weight_decay=0.001,
        optim="adamw_8bit",
        seed=3407,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Use Unsloth's train_on_responses_only to mask prompt tokens,
    # so the model only learns from the assistant response.
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("=== RCTP dataset sample ===")
    for k, v in train_dataset[0].items():
        print(f"  {k}: {str(v)[:200]}")
    print("===========================")

    SAMPLE_LOG.info("stage=pre_stage1_train type=summary trajectories=%d epochs=%d batch_size=%d",
                    len(rctp_trajectories), training_args.num_train_epochs, training_args.per_device_train_batch_size)
    trainer.train()
    trainer.save_model(rctp_cfg["output_dir"])
    tokenizer.save_pretrained(Path(rctp_cfg["output_dir"]))
    print(f"[Stage 1] RCTP-FT checkpoint saved -> {rctp_cfg['output_dir']}")
    print("(This checkpoint becomes pi_ref for Stage 2, per Algorithm 1 line 9.)")
```

```python
!zip -r outputs/rctp_ft_model.zip outputs/rctp_ft_model
from IPython.display import FileLink
FileLink("outputs/rctp_ft_model.zip")
```

```python
# ===================== Unified model loader for Stage 2 =====================
# Loads a base model with a fine-tuned LoRA adapter for inference or training,
# controlled by the `mode` argument.  Named parameters base_model_path /
# adapter_model_path make it clear what each path refers to.
# (Now handled by the unified load_model() above.)

```

```python
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # [NEW] Extra 30% context lengths!
if MODE in ("grpo", "rc_grpo"):
    # ===================== STAGE 2: RC-GRPO (or vanilla GRPO baseline) =====================
    print("\n" + "=" * 70)
    print(f"STAGE 2: {'RC-GRPO' if MODE == 'rc_grpo' else 'Vanilla GRPO baseline'}")
    print("=" * 70)

    # ── Set model paths for Stage 2 ──────────────────────────────────
    # The adapter checkpoint is typically the Stage 1 (RCTP-FT) output.
    # Adjust these to point to your trained checkpoint directories.
    base_model_path = "unsloth/Qwen3-4B-Instruct-2507"
    adapter_model_path = TRAIN_CONFIG["rctp_ft"]["output_dir"]  # Stage 1 output

    print(f"[Stage 2] Loading base_model_path = {base_model_path}")
    print(f"[Stage 2] Loading adapter_model_path = {adapter_model_path}")

    model, tokenizer = load_model(
        base_model_name=base_model_path,
        adapter_model_path=adapter_model_path,
        mode="train",          # keep in train mode for further fine-tuning
        fast_inference=True,
        max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],
        load_in_4bit=False,
        env_name=ENV_NAME,
    )

    from datasets import Dataset
    dataset = Dataset.from_list(load_jsonl(TRAIN_CONFIG["data"]["grpo_path"]))

    grpo_args = build_grpo_config(TRAIN_CONFIG, output_dir=TRAIN_CONFIG["training"]["output_dir"])

    if MODE == "rc_grpo":
        trainer = RCGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=build_trl_reward_functions("rc_grpo"),
            args=grpo_args,
            train_dataset=dataset,
            high_reward_probability=high_reward_probability,  # from Stage 1 stats, Eq. 3
        )
    elif MODE == "grpo":
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=build_trl_reward_functions("grpo"),
            args=grpo_args,
            train_dataset=dataset,
        )

    print(f"Trainer initialized for {MODE}")
    print("=== GRPO dataset sample ===")
    for k, v in dataset[0].items():
        print(f"  {k}: {str(v)[:200]}")
    print("===========================")

    custom_tags = ["<reasoning>", "</reasoning>", "<tool_call>", "</tool_call>", HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN]
    existing = set(tokenizer.all_special_tokens)
    conflicts = [tag for tag in custom_tags if tag in existing and tag not in REWARD_TOKENS]
    if conflicts:
        print(f"WARNING: custom tags collide with existing special tokens: {conflicts}")
    else:
        print("No conflicts detected with custom tags.")
    SAMPLE_LOG.info("stage=pre_stage2_train type=summary algorithm=%s dataset_size=%d generations=%d",
                    MODE, len(dataset), grpo_args.num_generations)
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # ── Auto-save model and tokenizer after training ──
    output_dir = MODE_OUTPUT_DIR
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    !zip -r {output_dir}.zip {output_dir}
    from IPython.display import FileLink
    FileLink(f"{output_dir}.zip")
else:
    print(f"Skipping Stage 2 (MODE={MODE}).")

```

```python
# ── Colab eval: download model from Hugging Face Hub to MODE_OUTPUT_DIR ──
# The HF_TOKEN secret must be set in Colab secrets for private repos.
COLAB_EVAL_MODEL_ID = "dzungpham/telecom-toolcaller"
if ENV_NAME == "colab":
    from huggingface_hub import snapshot_download
    os.makedirs(MODE_OUTPUT_DIR, exist_ok=True)
    print(f"[Colab] Downloading model {COLAB_EVAL_MODEL_ID} -> {MODE_OUTPUT_DIR}")
    snapshot_download(
        repo_id=COLAB_EVAL_MODEL_ID,
        local_dir=MODE_OUTPUT_DIR,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    print("[Colab] Download complete.")
```

```python
# ===================== EVALUATION =====================
test_dataset_path = TRAIN_CONFIG["data"]["test_path"]
if Path(test_dataset_path).exists():
    retriever = FunctionRetriever(function_library, method="hybrid")
    sandbox = Sandbox(function_library)

    if EVAL_USE_VLLM:
        eval_result = evaluate_model_vllm(
            model_path=MODE_OUTPUT_DIR,
            test_dataset_path=test_dataset_path,
            function_library=function_library,
            retriever=retriever,
            sandbox=sandbox,
            top_k=5,
            max_new_tokens=512,
            model_name_tag=MODE,
            condition_on_high_reward=(MODE == "rc_grpo"),
            argument_values=argument_values_catalog,
            batch_size=TRAIN_CONFIG["data"].get("eval_batch_size", 32),
        )
    else:
        eval_result = evaluate_model(
            model_path=MODE_OUTPUT_DIR,
            test_dataset_path=test_dataset_path,
            function_library=function_library,
            retriever=retriever,
            sandbox=sandbox,
            top_k=5,
            max_new_tokens=512,
            model_name_tag=MODE,
            condition_on_high_reward=(MODE == "rc_grpo"),
            argument_values=argument_values_catalog,
        )
    from tabulate import tabulate
    agg = eval_result["aggregate"]
    metric_names = [
        ("Function Selection Accuracy", "function_selection_accuracy"),
        ("Argument Accuracy", "argument_accuracy"),
        ("Schema Validity", "schema_validity"),
        ("Execution Success Rate", "execution_success_rate"),
        ("Task Success Rate", "task_success_rate"),
        ("Hallucinated Call Rate", "hallucinated_call_rate"),
        ("Abstention Accuracy", "abstention_accuracy"),
        ("Latency (ms)", "latency_ms"),
        ("Cost / Query (USD)", "cost_per_query_usd"),
    ]
    rows = []
    for display_name, key in metric_names:
        mean_val = agg.get(key, float("nan"))
        std_val = agg.get(f"{key}__std", float("nan"))
        import math
        if isinstance(std_val, (int, float)) and not math.isnan(std_val):
            rows.append([display_name, f"{mean_val:.4f} \u00b1 {std_val:.4f}"])
        else:
            rows.append([display_name, f"{mean_val:.4f}"])
    print(f"\nEvaluation Benchmark \u2014 {MODE}")
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="grid"))
    generate_report([eval_result])
    !zip -r {MODE_OUTPUT_DIR}.zip {MODE_OUTPUT_DIR}
    from IPython.display import FileLink
    FileLink(f"{MODE_OUTPUT_DIR}.zip")
else:
    print("Test dataset not found; skipping evaluation.")

```

```python
del model
del tokenizer
import gc
gc.collect()
torch.cuda.empty_cache()
```
