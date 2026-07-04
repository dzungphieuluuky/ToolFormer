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
# ── Base model path (auto-detects Kaggle path) ────────────────────────
if ENV_NAME == "kaggle":
    BASE_MODEL_PATH = "/kaggle/input/models/dzung271828/unsloth/transformers/default/4/qwen3-4b-instruct-2507/qwen3-4b-instruct-2507"
else:
    BASE_MODEL_PATH = "unsloth/Qwen3-4B-Instruct-2507"
print(f"Base model: {BASE_MODEL_PATH}")
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
    !pip install --no-index --find-links=/kaggle/input/datasets/nctuan/nvidia-offline-packages-nemotron/ /kaggle/input/datasets/nctuan/nvidia-offline-packages-nemotron/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

```python _cell_guid="version-check-001" jupyter={"outputs_hidden": false}
#@title Check installed library versions
import sys
import importlib

_LIBS = {
    "unsloth": None,
    "unsloth_zoo": None,
    "torch": None,
    "transformers": None,
    "trl": None,
    "peft": None,
    "vllm": None,
    "datasets": None,
    "accelerate": None,
    "bitsandbytes": None,
    "flash_attn": None,
}

print(f"{'Library':<20} {'Version':<20}")
print("-" * 40)
for lib_name in _LIBS:
    try:
        mod = importlib.import_module(lib_name)
        ver = getattr(mod, "__version__", "no __version__")
        print(f"{lib_name:<20} {ver:<20}")
    except ImportError:
        print(f"{lib_name:<20} {'not installed':<20}")
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
import warnings
import logging
import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass, field

import unsloth
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

def load_function_library(library_path: str) -> dict:
    with open(library_path, "r", encoding="utf-8") as fh:
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
        fmt = 0.0
        if has_tool_call:
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

    Wraps the continuous ``compute_action_coverage_reward`` and thresholds
    at >= 1.0 for backward compatibility with code that expects a binary
    signal.  1.0 means every gold call and every parameter was matched
    exactly — the same semantics as the old all-or-nothing check.

    Args:
        ground_truth: Ground-truth dict with a ``"calls"`` key.
        function_library: Dict mapping function names to schemas (unused but
                          kept for API compatibility).

    Returns:
        A callable ``(response_text: str) -> int`` that returns 0 or 1.
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
        # Threshold the continuous score at 1.0 (perfect match).
        return 1 if compute_action_coverage_reward(agent_calls, gold_calls) >= 1.0 else 0

    return _reward


# ─────────────────────────────────────────────────────────────────────
# TRL-compatible batch reward functions
# ─────────────────────────────────────────────────────────────────────

def function_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """F1-based function selection reward, continuous in [0, 1].

    Computes the F1 score between the set of gold function names and the set of
    agent-called function names.  This penalises both missing calls (low recall)
    AND spurious extra calls (low precision), unlike the previous binary check
    which only enforced recall.

    Args:
        completions: List of model response strings.
        ground_truth: List of ground-truth dicts (or JSON strings), each with a
                      ``"calls"`` key containing ``{"function": ..., "arguments": ...}``.
        **kwargs: Additional keyword arguments (ignored; for TRL compatibility).

    Returns:
        List of float scores in [0.0, 1.0].  A correct abstention (no gold calls
        and no agent calls) yields 1.0.
    """
    rewards = []
    for c, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        gold_calls = gt.get("calls", [])
        gold_funcs = [gc["function"] for gc in gold_calls if isinstance(gc, dict)]
        agent_calls = extract_all_calls(c)
        agent_funcs = [ac.get("function") for ac in agent_calls if isinstance(ac, dict)]

        # Abstention: no gold calls expected and none produced.
        if not gold_funcs:
            rewards.append(1.0 if not agent_funcs else 0.0)
            continue

        if not agent_funcs:
            rewards.append(0.0)
            continue

        # Precision = fraction of agent calls that are gold calls.
        true_positives = sum(1 for af in agent_funcs if af in gold_funcs)
        precision = true_positives / len(agent_funcs)

        # Recall = fraction of gold calls found among agent calls.
        recall = sum(1 for gf in gold_funcs if gf in agent_funcs) / len(gold_funcs)

        # F1 = harmonic mean of precision and recall.
        if precision + recall == 0.0:
            rewards.append(0.0)
        else:
            rewards.append(2.0 * precision * recall / (precision + recall))
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Multi-component format reward, continuous in [0, 1].

    Composed of three equally-weighted sub-checks to close trivial-hack
    vectors (e.g. inserting a bare ``<tool_call>`` token with no real call):

      1. Tag presence (0.3):  whether ``<tool_call>`` and ``</tool_call>`` tags appear.
      2. JSON parseability (0.3): whether extracted tool calls are valid JSON.
      3. Clean output (0.4): whether the response contains no text outside
         ``<tool_call>`` / ``<reasoning>`` blocks (i.e. no "garbage" tokens).

    Args:
        completions: List of model response strings.
        **kwargs: Additional keyword arguments (ignored; for TRL compatibility).

    Returns:
        List of float scores in [0.0, 1.0].
    """
    rewards = []
    for c in completions:
        # 1. Tag presence (0.3)
        has_open = bool(_TOOL_CALL_RE.search(c))
        has_close = "</tool_call>" in c
        tag_score = 1.0 if (has_open and has_close) else 0.0

        # 2. JSON parseability (0.3) — every extracted call must parse.
        calls = extract_all_calls(c) if has_open else []
        json_score = 1.0 if calls else 0.0
        if calls:
            # All must be valid dicts with "function" key.
            json_score = 1.0 if all(
                isinstance(ac, dict) and "function" in ac for ac in calls
            ) else 0.0

        # 3. Clean output (0.4) — no text outside <tool_call> / <reasoning>.
        #    Strip both block types; any non-whitespace remaining is "garbage".
        cleaned = _TOOL_CALL_RE.sub("", c)
        cleaned = _REASONING_RE.sub("", cleaned)
        cleaned = cleaned.strip()
        clean_score = 1.0 if not cleaned else 0.0

        # Weighted composite
        score = 0.3 * tag_score + 0.3 * json_score + 0.4 * clean_score
        rewards.append(score)
    return rewards

def argument_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """Continuous argument-accuracy reward in [0, 1].

    Delegates to ``compute_action_coverage_reward`` which returns a float
    based on per-parameter matching accuracy.  Removes the previous
    ``float()`` cast since that function now returns float natively.

    Args:
        completions: List of model response strings.
        ground_truth: List of ground-truth dicts (or JSON strings).
        **kwargs: Additional keyword arguments (ignored; for TRL compatibility).

    Returns:
        List of float scores in [0.0, 1.0].
    """
    rewards = []
    for c, gt_raw in zip(completions, ground_truth):
        gt = _parse_gt(gt_raw)
        gold_calls = [
            {"function": c["function"], "arguments": c.get("arguments", {})}
            for c in gt.get("calls", [])
        ]
        if not gold_calls:
            agent_calls = extract_all_calls(c)
            rewards.append(1.0 if not agent_calls else 0.0)
            continue
        agent_calls = extract_all_calls(c)
        if not agent_calls:
            rewards.append(0.0)
            continue
        rewards.append(compute_action_coverage_reward(agent_calls, gold_calls))
    return rewards


def composite_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """TRL-compatible: weighted sum of format + function + arguments (no reasoning term per Eq. 5)."""
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
                0.10 * (1.0 if bool(_TOOL_CALL_RE.search(c)) else 0.0)
                + 0.30 * func_selection_ok(c, expected_func)
                + 0.40 * args_accuracy(c, expected_args)
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
) -> float:
    """Continuous action-coverage reward in [0, 1] based on per-parameter accuracy.

    For each gold call, finds the matching agent call by function name and scores
    the fraction of gold parameters that the agent correctly predicts (with type
    coercion).  Returns the mean across all gold calls.

    Args:
        agent_tool_calls: List of agent-produced call dicts, each with
                          ``"function"`` and ``"arguments"`` keys.
        gold_tool_calls: List of ground-truth call dicts.

    Returns:
        Float in [0.0, 1.0].  1.0 when every gold call is fully covered (all
        parameters match exactly).  0.0 when no gold call is matched at all.
    """

    def _param_match_score(agent_args: dict, gold_args: dict) -> float:
        """Fraction of gold parameters matched (with type coercion and strip)."""
        if not gold_args:
            return 1.0
        matched = 0
        for param_key, param_val in gold_args.items():
            agent_val = agent_args.get(param_key)
            if agent_val is None:
                continue
            if str(agent_val).strip() == str(param_val).strip():
                matched += 1
        return matched / len(gold_args)

    if not gold_tool_calls:
        return 1.0
    if not agent_tool_calls:
        return 0.0

    call_scores = []
    for gold_call in gold_tool_calls:
        gold_func = gold_call.get("function")
        gold_args = gold_call.get("arguments", {})

        # Find matching agent call by function name.
        matching_agent = None
        for ac in agent_tool_calls:
            if ac.get("function") == gold_func:
                matching_agent = ac
                break

        if matching_agent is None:
            call_scores.append(0.0)
        else:
            agent_args = matching_agent.get("arguments", {})
            call_scores.append(_param_match_score(agent_args, gold_args))

    return sum(call_scores) / len(call_scores)
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
from torch.distributed import gather_object, broadcast_object_list
from contextlib import nullcontext


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


# ─────────────────────────────────────────────────────────────────────
# MMR diversity reweighting functions (from WeiKangda/MMR-GRPO, ACL 2026)
# ─────────────────────────────────────────────────────────────────────


def mmr_reweight_original(rewards, embeddings, lambda_div=0.7):
    """
    Original MMR: greedy selection with fixed λ.
    rewards: (B,)
    embeddings: (B, D), L2-normalized
    Returns adjusted rewards in original order.
    """
    N = rewards.size(0)
    device = rewards.device
    sim_matrix = embeddings @ embeddings.T
    sim_matrix = sim_matrix - torch.eye(N, device=device) * 1e9

    adjusted = torch.zeros_like(rewards)
    best_sim = torch.zeros(N, device=device)
    selected = torch.zeros(N, dtype=torch.bool, device=device)

    first = torch.argmax(rewards)
    selected[first] = True
    adjusted[first] = rewards[first]

    for _ in range(N - 1):
        best_sim = torch.max(best_sim, sim_matrix[:, selected].max(dim=1).values)
        scores = lambda_div * rewards - (1.0 - lambda_div) * best_sim
        scores[selected] = -1e9
        next_idx = torch.argmax(scores)
        selected[next_idx] = True
        adjusted[next_idx] = scores[next_idx]

    return adjusted


def mmr_reweight_std(rewards, embeddings, temp=2.0, eps=1e-8):
    """
    Adaptive λ from reward std: λ = rel_std_scaled / (1 + rel_std_scaled).
    Higher std → more diversity emphasis.
    Returns (adjusted_rewards, lambda_used).
    """
    N = rewards.size(0)
    device = rewards.device

    std = rewards.std(unbiased=False)
    mean = rewards.mean()
    rel_std = std / (mean.abs() + eps)
    rel_std_scaled = rel_std * temp
    lambda_div = rel_std_scaled / (1.0 + rel_std_scaled)
    lambda_used = float(lambda_div)

    sim_matrix = embeddings @ embeddings.T
    sim_matrix = sim_matrix - torch.eye(N, device=device) * 1e9
    adjusted = torch.zeros_like(rewards)
    best_sim = torch.zeros(N, device=device)
    selected = torch.zeros(N, dtype=torch.bool, device=device)

    first = torch.argmax(rewards)
    selected[first] = True
    adjusted[first] = rewards[first]

    lam = lambda_used
    for _ in range(N - 1):
        best_sim = torch.max(best_sim, sim_matrix[:, selected].max(dim=1).values)
        scores = lam * rewards - (1.0 - lam) * best_sim
        scores[selected] = -1e9
        next_idx = torch.argmax(scores)
        selected[next_idx] = True
        adjusted[next_idx] = scores[next_idx]

    # Rescale to match original reward scale
    adj_mean = adjusted.mean()
    adj_std = adjusted.std(unbiased=False)
    if adj_std.item() > eps:
        adjusted = (adjusted - adj_mean) / (adj_std + eps)
        adjusted = adjusted * (std + eps) + mean

    return adjusted, lambda_used


def mmr_reweight_sigmoid(rewards, embeddings, eps=1e-8):
    """
    Adaptive λ via sigmoid(std): λ = σ(std). No hyperparameters.
    Smooth, scale-invariant adaptation.
    Returns (adjusted_rewards, lambda_used).
    """
    N = rewards.size(0)
    device = rewards.device

    lambda_div = torch.sigmoid(rewards.std(unbiased=False))

    sim_matrix = embeddings @ embeddings.T
    sim_matrix = sim_matrix - torch.eye(N, device=device) * 1e9
    adjusted = torch.zeros_like(rewards)
    best_sim = torch.zeros(N, device=device)
    selected = torch.zeros(N, dtype=torch.bool, device=device)

    first = torch.argmax(rewards)
    selected[first] = True
    adjusted[first] = rewards[first]

    for _ in range(N - 1):
        best_sim = torch.max(best_sim, sim_matrix[:, selected].max(dim=1).values)
        scores = lambda_div * rewards - (1.0 - lambda_div) * best_sim
        scores[selected] = -1e9
        next_idx = torch.argmax(scores)
        selected[next_idx] = True
        adjusted[next_idx] = scores[next_idx]

    return adjusted, lambda_div.item()


# ─────────────────────────────────────────────────────────────────────
# AVSPO helper functions (from He et al., ICML 2026, arXiv:2605.21125)
# ─────────────────────────────────────────────────────────────────────


def compute_acr(rewards, G: int, tau: float = 1e-6):
    """
    Compute the Advantage Collapse Rate (Definition 4.1).

    ACR = fraction of groups where the within-group reward std < τ.
    Higher ACR → more groups are collapsed → more virtual samples needed.

    Args:
        rewards: (B,) tensor of raw rewards, where B = num_groups * G.
        G: Number of generations per prompt (num_generations).
        tau: Collapse detection threshold (default: 1e-6, the paper's τ_collapse).

    Returns:
        (scalar_acr, collapse_flags): ACR as a scalar tensor, and a boolean
        tensor of shape (num_groups,) indicating which groups are collapsed.
    """
    group_rewards = rewards.view(-1, G)                     # (num_groups, G)
    group_std = group_rewards.std(dim=1, unbiased=False)    # (num_groups,)
    collapse_flags = group_std < tau                        # (num_groups,) bool
    num_groups = group_rewards.size(0)
    acr = collapse_flags.float().mean()                     # scalar
    return acr, collapse_flags


def generate_virtual_samples(
    K_desired: int,
    G: int,
    is_all_correct: bool,
    max_possible_reward: float,
    anchor_reward: float = 0.1,
):
    """
    Generate K virtual reward values for a collapsed group (Eq. 9, adapted
    for continuous rewards).

    K = max(1, min(K_desired, G)) — bounded by G and at least 1.
    Virtual rewards are generated following the stratified assignment formula:
    for j in 1..K: r_j = r_anchor_descending * (K+1-j) / (K+1)

    If the group is all-correct (r_obs ≈ max_possible_reward): virtual rewards
    descend from max_possible_reward (e.g., [3.0*K/(K+1), 3.0*(K-1)/(K+1), ...]).

    If the group is all-incorrect (r_obs ≈ 0): virtual rewards descend from
    anchor_reward * max_possible_reward (e.g., [0.3*K/(K+1), 0.3*(K-1)/(K+1), ...]).

    Args:
        K_desired: Desired number of virtual samples (before bounding).
        G: Number of real samples in the group.
        is_all_correct: Whether the group's mean reward is above the midpoint.
        max_possible_reward: Maximum possible reward (e.g., 3.0 for 3 funcs).
        anchor_reward: Scale for all-incorrect virtual rewards (default: 0.1).

    Returns:
        (K, virtual_rewards): K (actual count) and a 1D tensor of K virtual rewards.
    """
    K = max(1, min(K_desired, G))
    max_r = max_possible_reward
    anchor_r = anchor_reward * max_r

    if is_all_correct:
        # Descend from max_possible_reward
        top = max_r
    else:
        # Descend from anchor_reward * max_possible_reward
        top = anchor_r

    # Stratified assignment: r_j = top * (K+1-j) / (K+1) for j in 1..K
    j = torch.arange(1, K + 1, dtype=torch.float32)
    virtual_rewards = top * (K + 1 - j) / (K + 1)
    return K, virtual_rewards


def adapt_tau(
    tau_current: float,
    acr: float,
    delta_J: float,
    eta: float = 0.01,
) -> float:
    """
    Adapt the collapse detection threshold τ_adapt (Eq. 10).

    τ_adapt := τ_adapt + η * (ACR - delta_J)
    where delta_J = current_mean_raw_reward - previous_mean_raw_reward.

    When rewards improve (delta_J > 0), ACR is compared against the
    improvement — if ACR exceeds improvement, threshold increases
    (more groups flagged) to inject more virtual samples for harder groups.

    Args:
        tau_current: Current τ_adapt value.
        acr: ACR of the current batch (scalar float).
        delta_J: Change in mean reward from previous batch.
        eta: Learning rate for adaptation (default: 0.01, paper's η).

    Returns:
        Updated τ_adapt (clipped to [1e-8, 1.0]).
    """
    new_tau = tau_current + eta * (acr - delta_J)
    return max(1e-8, min(1.0, new_tau))


# ─────────────────────────────────────────────────────────────────────
# GPU resource cleanup helper
# ─────────────────────────────────────────────────────────────────────


def cleanup_training(trainer=None, model=None, tokenizer=None, logger=None):
    """
    Release ALL training resources so evaluation starts with clean GPU state.

    vLLM reference: PR #46023 (shutdown + gc.unfreeze + cleanup_dist_env)
    vLLM issue #6544, #5716: del llm alone is insufficient
    Unsloth docs: UNSLOTH_VLLM_STANDBY=0 during eval prevents dual engines

    Safe to call multiple times (idempotent via try/except per step).
    """
    _log = logger.info if hasattr(logger, 'info') else print

    # ── 1. Shutdown vLLM engine (if trainer has one) ────────────────
    if trainer is not None and hasattr(trainer, 'llm') and trainer.llm is not None:
        try:
            llm = trainer.llm
            # Try the modern vLLM shutdown() path (added in PR #46023, vLLM ~v0.17+)
            if hasattr(llm, 'shutdown') and callable(llm.shutdown):
                llm.shutdown()
                _log("[cleanup] vLLM LLM.shutdown() completed")
            else:
                # Fallback: sleep level=2 (discard weights + KV cache)
                if hasattr(llm, 'sleep') and callable(llm.sleep):
                    llm.sleep(level=2)
                    _log("[cleanup] vLLM sleep(level=2) completed")
        except Exception as exc:
            _log(f"[cleanup] WARNING: vLLM shutdown failed: {exc}")

    # ── 2. Destroy vLLM distributed state ────────────────────────────
    # This is critical — without it, vLLM's internal references keep GPU memory
    try:
        from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()
    except ImportError:
        try:
            # Older vLLM (<0.12) had this in a different location
            from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except ImportError:
            pass  # Not running distributed or very old vLLM
    except Exception as exc:
        _log(f"[cleanup] WARNING: distributed teardown failed: {exc}")

    # ── 3. Destroy torch distributed process group ───────────────────
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as exc:
        _log(f"[cleanup] WARNING: destroy_process_group failed: {exc}")

    # ── 4. Unfreeze GC heap (vLLM V1 freezes startup objects) ───────
    # Without this, gc.collect() cannot reclaim vLLM engine objects
    try:
        import gc as _gc
        _gc.unfreeze()
    except Exception:
        pass  # Not frozen or old Python

    # ── 5. Delete trainer ────────────────────────────────────────────
    if trainer is not None:
        try:
            # Try to release trainer's internal model reference first
            if hasattr(trainer, 'model') and trainer.model is not None:
                if hasattr(trainer.model, '_delete_llm'):
                    trainer.model._delete_llm()
        except Exception:
            pass
        try:
            del trainer
        except Exception:
            pass

    # ── 6. Delete model and tokenizer ────────────────────────────────
    if model is not None:
        try:
            # For Unsloth's FastLanguageModel, try internal cleanup
            if hasattr(model, 'delete_llm'):
                model.delete_llm()
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
    if tokenizer is not None:
        try:
            del tokenizer
        except Exception:
            pass

    # ── 7. Garbage collection ───────────────────────────────────────
    import gc as _gc2
    for _ in range(3):  # Triple-collect to clear reference cycles
        _gc2.collect()

    # ── 8. CUDA cache and allocator reset ────────────────────────────
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _log(f"[cleanup] GPU memory freed. "
         f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
         f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")


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

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """
        Backward-compat: TRL v0.24.0 renamed _get_per_token_logps to
        _get_per_token_logps_and_entropies (returns tuple[logps, entropies]).
        This wrapper provides the old single-tensor interface so all callers
        (MMRGRPOTrainer, AVSPOTrainer) continue to work unchanged.
        """
        logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep,
            compute_entropy=False,
        )
        return logps

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


class MMRGRPOTrainer(RCGRPOTrainer):
    """
    MMR-GRPO: Reward-conditioned GRPO + diversity-aware reward reweighting.

    Subclasses RCGRPOTrainer. Overrides `_generate_and_score_completions` to
    apply MMR reweighting to raw rewards BEFORE advantage computation,
    exactly as specified in Algorithm 1 of the MMR-GRPO paper.

    Embeddings are computed from the model's own hidden states (last layer,
    mean-pooled over completion tokens) — no external embedding model needed.
    This is both zero-dependency and policy-adaptive: representations evolve
    as the model fine-tunes.

    Reference: https://github.com/WeiKangda/MMR-GRPO (ACL 2026 Findings)
    TRL version targeted: v0.22.2
    """

    def __init__(self, *args, mmr_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mmr_config = mmr_config or {}
        self._mmr_log = []  # per-step MMR diagnostics
        # TRL v0.14.0 set self.sampling_params in GRPOTrainer.__init__, but v0.23.1+ removed it.
        # Reconstruct from individual args attributes (matches build_grpo_config defaults).
        self.sampling_params = SamplingParams(
            temperature=self.args.temperature,
            max_tokens=self.args.max_completion_length,
            top_p=getattr(self.args, 'top_p', 0.95),
            min_p=getattr(self.args, 'min_p', 0.05),
            seed=getattr(self.args, 'seed', 3407),
            stop=["</tool_call>"],
            include_stop_str_in_output=True,
        )

    @staticmethod
    def _render_prompts(inputs, processing_class):
        """
        Replicates TRL's maybe_apply_chat_template without importing from trl.data_utils.
        Handles both text (standard) and message-list (conversational) prompts.
        """
        prompts_text = []
        for example in inputs:
            prompt = example["prompt"]
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
                prompts_text.append(processing_class.apply_chat_template(prompt, tokenize=False))
            else:
                prompts_text.append(prompt)
        return prompts_text

    def _generate_and_score_completions(self, inputs):
        """
        Full override of TRL's generation-scoring pipeline with MMR insertion.

        Cannot call super() because advantages would be computed from un-adjusted
        rewards inside the parent. Instead, reproduce the parent's logic with MMR
        inserted between reward and advantage computation.

        TRL v0.22.2 API calls used:
        - self._calculate_rewards(inputs, prompts, completions, completion_ids.tolist())
        - self._move_model_to_vllm()
        - self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        - self._get_per_token_logps(self.model, prompt_completion_ids, prompt_mask, logits_to_keep)
        """
        device = self.accelerator.device

        # Step 1: Get prompts
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self._render_prompts(inputs, self.processing_class)
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        )
        prompt_inputs = {k: v.to(self.accelerator.device) for k, v in prompt_inputs.items()}
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Step 2: Generate completions
        if self.args.use_vllm:
            # vLLM colocate path (trl v0.22.2: self.llm.generate)
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            _world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if _world_size > 1:
                if self.accelerator.is_main_process:
                    _gather_list = [None] * _world_size
                    gather_object(prompts_text, object_gather_list=_gather_list)
                    all_prompts_text = [item for sublist in _gather_list for item in sublist]
                else:
                    gather_object(prompts_text)
                    all_prompts_text = prompts_text
            else:
                all_prompts_text = prompts_text

            if self.accelerator.is_main_process:
                ordered_set_of_prompts = all_prompts_text[::self.num_generations]
                outputs = self.llm.generate(
                    ordered_set_of_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                completion_ids = [
                    out.token_ids
                    for completions in outputs
                    for out in completions.outputs
                ]
            else:
                completion_ids = [None] * len(all_prompts_text)

            if _world_size > 1:
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = torch.nn.utils.rnn.pad_sequence(
                completion_ids, batch_first=True,
                padding_value=self.processing_class.pad_token_id,
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Transformers generation path
            prompt_completion_ids = self.model.generate(
                prompt_ids, attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )

        # Extract completion_ids and completion_mask
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_mask = (completion_ids != self.processing_class.pad_token_id).float()

        # Decode for reward functions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Step 3: Compute raw rewards
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids.tolist()
        )
        rewards = rewards_per_func.sum(dim=1)

        # Save raw rewards for diagnostics
        raw_rewards_for_log = rewards.detach().clone()

        # Step 3.5: Apply MMR to rewards (THE ADDITION)
        if self.mmr_config.get("enabled", False):
            try:
                # Compute embeddings from model's OWN hidden states
                with torch.no_grad():
                    model_outputs = self.model(
                        input_ids=completion_ids,
                        attention_mask=completion_mask,
                        output_hidden_states=True,
                    )
                    # Last hidden layer: (B, T, D)
                    hidden = model_outputs.hidden_states[-1]
                    # Mean-pool over completion tokens
                    mask_3d = completion_mask.unsqueeze(-1)
                    embeddings = (hidden * mask_3d).sum(dim=1) / \
                        mask_3d.sum(dim=1).clamp(min=1)
                    # L2 normalize
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Apply selected MMR variant
                mode = self.mmr_config.get("mode", "sigmoid")
                if mode == "original":
                    rewards = mmr_reweight_original(
                        rewards, embeddings,
                        lambda_div=self.mmr_config.get("lambda_div", 0.7),
                    )
                elif mode == "std":
                    rewards, lambda_used = mmr_reweight_std(
                        rewards, embeddings,
                        temp=self.mmr_config.get("mmr_std_temp", 2.0),
                    )
                else:  # "sigmoid" (default, zero hyperparams)
                    rewards, lambda_used = mmr_reweight_sigmoid(rewards, embeddings)

                # Log MMR diagnostics
                with torch.no_grad():
                    sim_matrix = embeddings @ embeddings.T
                    mean_sim = sim_matrix.mean().item()
                self._mmr_log.append({
                    "step": self.state.global_step,
                    "mean_similarity": mean_sim,
                    "mode": mode,
                    "lambda": lambda_used if mode != "original" \
                        else self.mmr_config.get("lambda_div", 0.7),
                    "raw_reward_mean": raw_rewards_for_log.mean().item(),
                    "mmr_reward_mean": rewards.mean().item(),
                    "n_completions": len(rewards),
                })
            except Exception as exc:
                print(f"[MMR] WARNING: MMR failed ({exc}), using raw rewards.")

        # Step 4: Compute advantages (from potentially MMR-adjusted rewards)
        advantages = rewards.view(-1, self.num_generations)
        advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / \
                     (advantages.std(dim=1, keepdim=True) + 1e-4)
        advantages = advantages.view(-1)

        # Step 5: Compute old_per_token_logps (importance sampling)
        with torch.no_grad():
            old_per_token_logps = self._get_per_token_logps(
                self.model, prompt_completion_ids, prompt_mask,
                prompt_completion_ids.size(1) - prompt_ids.size(1),
            )

        # Step 6: Compute ref_per_token_logps (KL penalty)
        ref_per_token_logps = None
        beta = getattr(self.args, "beta", 0.0)
        if beta != 0.0:
            if self.ref_model is not None:
                ref_model = self.ref_model
                cm = nullcontext()
            elif hasattr(self.model, "disable_adapter"):
                ref_model = self.model
                cm = self.model.disable_adapter()
            else:
                ref_model = self.model
                cm = nullcontext()
            with torch.no_grad(), cm:
                ref_per_token_logps = self._get_per_token_logps(
                    ref_model, prompt_completion_ids, prompt_mask,
                    prompt_completion_ids.size(1) - prompt_ids.size(1),
                )

        # Step 7: Return dict (EXACT format expected by _compute_loss)
        num_items_in_batch = completion_mask.sum()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }


class GTPOTrainer(RCGRPOTrainer):
    """
    GTPO: Conflict-aware gradient correction (Simoni et al., arXiv 2508.03772).

    Inherits reward-conditioned rollout from RCGRPOTrainer and overrides
    `_compute_loss` with:
    - λ_{i,t} ∈ {0,1,2} conflict mask: tokens shared across correct and
      incorrect trajectories get λ=0 (shielded from contradictory gradients)
    - δ_i entropy filter: skips groups where mean entropy > threshold
    - Entropy regularization replacing KL: A' = A - γ·⟨H⟩
    """

    def __init__(self, *args, entropy_threshold: float = 0.5, gamma: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_threshold = entropy_threshold  # δ filter gate
        self.gamma = gamma  # entropy regularization coefficient

    def _compute_loss(self, model, inputs, num_items_in_batch=None):
        # ── 1. Forward pass: get current-policy logits ──────────────────
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :]  # Shift for next-token pred

        # ── 2. Compute per-token log-probs ─────────────────────────────
        # Reimplementation of TRL's internal helper for compat
        log_probs = logits.log_softmax(dim=-1)
        per_token_logps = log_probs.gather(
            dim=-1,
            index=inputs["input_ids"][:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        # ── 3. Get stored tensors ───────────────────────────────────────
        old_per_token_logps = inputs["old_per_token_logps"]       # (B, T)
        advantages = inputs["advantages"]                         # (B,)
        # completion_mask: 1 for completion tokens, 0 elsewhere
        completion_mask = inputs.get(
            "completion_mask",
            (inputs["input_ids"][:, 1:] != self.processing_class.pad_token_id).float(),
        )

        # ── 4. Log ratio and importance weights ───────────────────────
        log_ratio = per_token_logps - old_per_token_logps              # (B, T)
        ratios = torch.exp(log_ratio)                                  # (B, T)

        # ── 5. Shape into groups and classify ──────────────────────────
        G = self.args.num_generations
        B, T = log_ratio.shape
        num_groups = B // G

        adv_group = advantages.view(num_groups, G)                     # (num_groups, G)
        median_adv = adv_group.median(dim=1, keepdim=True).values      # per-group median
        is_high = (adv_group > median_adv).float()                     # 1 = high-reward traj
        is_high_flat = is_high.view(B)                                 # (B,)

        # ── 6. Build conflict mask λ_{i,t} via log-prob similarity ────
        # Tokens with similar log-probs in high- and low-advantage
        # trajectories are "shared" — shield them from contradictory gradients.
        # λ=0 (shared, shielded), λ=2 (high-only, protected), λ=1 (default).
        conflict_mask = torch.ones_like(ratios)                         # default λ=1
        for g in range(num_groups):
            grp_start = g * G
            grp_end = grp_start + G
            grp_logp = per_token_logps[grp_start:grp_end]               # (G, T)
            grp_is_high = is_high_flat[grp_start:grp_end]               # (G,)

            # Log-prob centroids for high- and low-advantage groups
            high_logp = grp_logp[grp_is_high > 0].mean(dim=0, keepdim=True)   # (1, T)
            low_logp = grp_logp[grp_is_high == 0].mean(dim=0, keepdim=True)   # (1, T)

            if high_logp.shape[0] == 0 or low_logp.shape[0] == 0:
                continue  # no conflict — all λ=1

            # Shared token: |logp_diff| < 0.1 nats across groups
            logp_diff = (high_logp - low_logp).abs()                     # (1, T)
            is_shared = logp_diff < 0.1                                  # (1, T)

            # High-only token: logp is > 0.1 nats higher in high-reward group
            high_logp_minus_low = (high_logp - low_logp)                  # (1, T)
            is_high_only = high_logp_minus_low > 0.1                     # (1, T)

            group_mask = torch.where(
                is_shared,
                torch.tensor(0.0, device=ratios.device),                 # λ=0: shielded
                torch.where(
                    is_high_only,
                    torch.tensor(2.0, device=ratios.device),             # λ=2: protected
                    torch.tensor(1.0, device=ratios.device),             # λ=1: normal
                ),
            )
            conflict_mask[grp_start:grp_end] = group_mask.expand_as(grp_logp)

        # ── 7. Entropy filter δ_i and entropy regularization ───────────
        probs = logits.softmax(dim=-1)                                 # (B, T-1, V)
        token_entropy = -(probs * probs.log()).sum(dim=-1)             # (B, T-1)

        # Match shape: pad or slice to T (if T-1 vs T mismatch)
        if token_entropy.shape[1] < T:
            # entropy is shorter than log_ratio — pad at the end
            pad = torch.zeros(B, T - token_entropy.shape[1], device=token_entropy.device)
            token_entropy_padded = torch.cat([token_entropy, pad], dim=1)
        else:
            token_entropy_padded = token_entropy[:, :T]

        # Mean entropy per trajectory (masked)
        mean_entropy = (token_entropy_padded * completion_mask).sum(dim=-1) / completion_mask.sum(dim=-1).clamp(min=1)  # (B,)

        # δ_i: skip uncertain trajectories
        entropy_filter = (mean_entropy < self.entropy_threshold).float()  # (B,)
        entropy_filter_expanded = entropy_filter.unsqueeze(-1)           # (B, 1)

        # Adjusted advantage: A'_i = A_i - γ·⟨H⟩_i
        adj_advantage = advantages - self.gamma * mean_entropy           # (B,)
        adj_adv_expanded = adj_advantage.unsqueeze(-1)                   # (B, 1)

        # ── 8. GTPO per-token loss ─────────────────────────────────────
        # J = (1/G) Σ_i (δ_i · (Â_i - γ⟨H⟩_i) / |o_i|) · Σ_t r_i,t · λ_i,t
        completion_lengths = completion_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)

        per_token_loss = -(
            entropy_filter_expanded
            * adj_adv_expanded
            / completion_lengths
            * ratios
            * conflict_mask
        )

        # ── 9. Aggregate (DAPO-style: sum / global active token count) ─
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if num_items_in_batch is not None:
            normalizer = num_items_in_batch / world_size
        else:
            # Fallback for TRL v0.22.2 which may not pass num_items_in_batch
            normalizer = completion_mask.sum() / world_size
        loss = (per_token_loss * completion_mask).sum() / normalizer

        return loss


# ─────────────────────────────────────────────────────────────────────
# AVSPOTrainer (from He et al., ICML 2026, arXiv:2605.21125)
# ─────────────────────────────────────────────────────────────────────


class AVSPOTrainer(RCGRPOTrainer):
    """
    AVSPO: Adaptive Virtual Sample Policy Optimization (He et al., ICML 2026).

    Inherits reward-conditioned rollout from RCGRPOTrainer. Overrides
    `_generate_and_score_completions` to:
    1. Inject reward tokens into prompts (same as RCGRPOTrainer, preserving RC)
    2. Generate completions (NO prompt deduplication — each slot has unique token)
    3. Compute raw rewards
    4. Detect collapsed groups (sigma_R < tau) and compute ACR
    5. When ACR > tau_adapt, inject virtual reward samples into collapsed groups
    6. Recompute advantages with per-group augmented statistics
    7. Return the standard dict (same format as parent)

    Virtual samples are synthetic reward values that participate ONLY in
    normalization statistics (mu, sigma) for advantage computation — they do
    NOT contribute to the policy gradient (no nabla_theta log pi_theta term
    exists for them).

    CAUTION: Generation uses NON-deduplicated prompts (all B, not [::G])
    because reward token injection creates DIFFERENT prompts per slot.
    Unlike MMRGRPOTrainer, [::G] deduplication would discard G-1 tokens.
    """

    def __init__(self, *args, avspo_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.avspo_config = avspo_config or {}
        self.tau_adapt = self.avspo_config.get("tau_adapt_initial", 0.5)
        self._prev_avg_reward = None
        self._acr_log = []  # per-step ACR diagnostics
        # max_possible_reward = number of reward functions (each returns [0,1], summed by TRL)
        self._max_possible_reward = float(len(self.reward_funcs)) if hasattr(self, "reward_funcs") else 3.0
        # TRL v0.14.0 set self.sampling_params in GRPOTrainer.__init__, but v0.23.1+ removed it.
        # Reconstruct from individual args attributes (matches build_grpo_config defaults).
        self.sampling_params = SamplingParams(
            temperature=self.args.temperature,
            max_tokens=self.args.max_completion_length,
            top_p=getattr(self.args, 'top_p', 0.95),
            min_p=getattr(self.args, 'min_p', 0.05),
            seed=getattr(self.args, 'seed', 3407),
            stop=["</tool_call>"],
            include_stop_str_in_output=True,
        )

    @staticmethod
    def _render_prompts(inputs, processing_class):
        """
        Replicates TRL's maybe_apply_chat_template without importing from trl.data_utils.
        Handles both text (standard) and message-list (conversational) prompts.
        (Same logic as MMRGRPOTrainer._render_prompts.)
        """
        prompts_text = []
        for example in inputs:
            prompt = example["prompt"]
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
                prompts_text.append(processing_class.apply_chat_template(prompt, tokenize=False))
            else:
                prompts_text.append(prompt)
        return prompts_text

    def _generate_and_score_completions(self, inputs):
        """
        Full override of TRL's generation-scoring pipeline with AVSPO insertion.

        Cannot call super()._generate_and_score_completions() because advantages
        would be computed from un-modified rewards inside the parent. Instead,
        reproduce the parent's logic with AVSPO inserted between reward and
        advantage computation.

        TRL v0.22.2 API calls used (same as MMRGRPOTrainer):
        - self._calculate_rewards(inputs, prompts, completions, completion_ids.tolist())
        - self._move_model_to_vllm()
        - self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        - self._get_per_token_logps(self.model, prompt_completion_ids, prompt_mask, logits_to_keep)
        """
        import math

        device = self.accelerator.device
        G = self.args.num_generations

        # ── Step 1: Reward token injection (same as RCGRPOTrainer) ────────
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
                conditioned_example["_reward_token"] = r_j
                conditioned_inputs.append(conditioned_example)
        inputs = conditioned_inputs

        # ── Step 2: Get prompts ──────────────────────────────────────────
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self._render_prompts(inputs, self.processing_class)
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        )
        prompt_inputs = {k: v.to(self.accelerator.device) for k, v in prompt_inputs.items()}
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # ── Step 3: Generate completions (NO vLLM deduplication!) ────────
        # CRITICAL: Unlike MMRGRPOTrainer, we do NOT deduplicate prompts
        # via all_prompts_text[::self.num_generations]. Each prompt has a
        # unique reward token — deduplication would discard G-1 tokens.
        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            _world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if _world_size > 1:
                if self.accelerator.is_main_process:
                    _gather_list = [None] * _world_size
                    gather_object(prompts_text, object_gather_list=_gather_list)
                    all_prompts_text = [item for sublist in _gather_list for item in sublist]
                else:
                    gather_object(prompts_text)
                    all_prompts_text = prompts_text
            else:
                all_prompts_text = prompts_text

            if self.accelerator.is_main_process:
                # Pass ALL B prompts (not [::G]) — each has a unique reward token
                outputs = self.llm.generate(
                    all_prompts_text,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                completion_ids = [
                    out.token_ids
                    for completions in outputs
                    for out in completions.outputs
                ]
            else:
                completion_ids = [None] * len(all_prompts_text)

            if _world_size > 1:
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = torch.nn.utils.rnn.pad_sequence(
                completion_ids, batch_first=True,
                padding_value=self.processing_class.pad_token_id,
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            prompt_completion_ids = self.model.generate(
                prompt_ids, attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )

        # Extract completion_ids and completion_mask
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_mask = (completion_ids != self.processing_class.pad_token_id).float()

        # Decode for reward functions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # ── Step 4: Compute raw rewards ──────────────────────────────────
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids.tolist()
        )
        rewards = rewards_per_func.sum(dim=1)

        # Save mean raw reward (BEFORE AVSPO) for delta_J tracking
        current_raw_mean = rewards.mean().item()

        # ── Step 5: Compute advantages with AVSPO virtual samples ────────
        num_groups = rewards.size(0) // G
        group_rewards = rewards.view(num_groups, G)

        # Compute ACR
        tau_collapse = self.avspo_config.get("acr_threshold", 1e-6)
        acr, collapse_flags = compute_acr(rewards, G, tau=tau_collapse)

        max_r = self._max_possible_reward
        anchor_r = self.avspo_config.get("anchor_reward", 0.1)
        alpha = self.avspo_config.get("alpha", 0.5)
        eps = 1e-8

        advantages = torch.zeros_like(rewards)
        total_k = 0  # total virtual samples injected (for logging)

        if acr > self.tau_adapt:
            # Virtual sample injection phase
            try:
                for g in range(num_groups):
                    if collapse_flags[g]:
                        # Determine direction using reward-scale-aware threshold
                        is_correct = group_rewards[g].mean() > (max_r / 2.0)
                        # Compute K from ACR
                        acr_val = acr.item() if hasattr(acr, "item") else float(acr)
                        K_desired = max(1, int(math.ceil(G * (acr_val ** alpha))))
                        K, virtual_rewards = generate_virtual_samples(
                            K_desired, G, is_correct, max_r,
                            anchor_reward=anchor_r,
                        )
                        total_k += K
                        # Augmented statistics: G real + K virtual
                        augmented_pool = torch.cat([
                            group_rewards[g],
                            virtual_rewards.to(group_rewards.device),
                        ])
                        aug_mean = augmented_pool.mean()
                        aug_std = augmented_pool.std(unbiased=False) + eps
                        # Normalize ONLY the G real rewards
                        advantages[g * G: (g + 1) * G] = \
                            (group_rewards[g] - aug_mean) / aug_std
                    else:
                        # Standard GRPO normalization
                        grp_mean = group_rewards[g].mean()
                        grp_std = group_rewards[g].std(unbiased=False) + eps
                        advantages[g * G: (g + 1) * G] = \
                            (group_rewards[g] - grp_mean) / grp_std
            except Exception as exc:
                print(f"[AVSPO] WARNING: virtual sample injection failed ({exc}), "
                      f"falling back to standard GRPO advantages.")
                advantages = rewards.view(-1, G)
                advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / \
                             (advantages.std(dim=1, keepdim=True) + eps)
                advantages = advantages.view(-1)
        else:
            # Standard GRPO normalization (no virtual samples needed)
            advantages = group_rewards - group_rewards.mean(dim=1, keepdim=True)
            advantages = advantages / (group_rewards.std(dim=1, keepdim=True, unbiased=False) + eps)
            advantages = advantages.view(-1)

        # ── Step 6: Update tau_adapt ─────────────────────────────────────
        delta_J = current_raw_mean - (self._prev_avg_reward if self._prev_avg_reward is not None else current_raw_mean)
        tau_eta = self.avspo_config.get("tau_adapt_eta", 0.01)
        self.tau_adapt = adapt_tau(self.tau_adapt, float(acr), delta_J, eta=tau_eta)
        self._prev_avg_reward = current_raw_mean

        # ── Step 7: ACR diagnostics logging ──────────────────────────────
        acr_log_interval = self.avspo_config.get("acr_log_interval", 10)
        if self.state.global_step % acr_log_interval == 0:
            self._acr_log.append({
                "step": self.state.global_step,
                "acr": acr.item() if hasattr(acr, "item") else float(acr),
                "tau_adapt": self.tau_adapt,
                "n_collapsed_groups": collapse_flags.sum().item() if hasattr(collapse_flags, "sum") else 0,
                "n_groups": num_groups,
                "virtual_samples_injected": total_k,
                "mean_raw_reward": current_raw_mean,
            })

        # ── Step 8: Compute old_per_token_logps ─────────────────────────
        with torch.no_grad():
            old_per_token_logps = self._get_per_token_logps(
                self.model, prompt_completion_ids, prompt_mask,
                prompt_completion_ids.size(1) - prompt_ids.size(1),
            )

        # ── Step 9: Compute ref_per_token_logps (KL penalty) ────────────
        ref_per_token_logps = None
        beta = getattr(self.args, "beta", 0.0)
        if beta != 0.0:
            if self.ref_model is not None:
                ref_model = self.ref_model
                cm = nullcontext()
            elif hasattr(self.model, "disable_adapter"):
                ref_model = self.model
                cm = self.model.disable_adapter()
            else:
                ref_model = self.model
                cm = nullcontext()
            with torch.no_grad(), cm:
                ref_per_token_logps = self._get_per_token_logps(
                    ref_model, prompt_completion_ids, prompt_mask,
                    prompt_completion_ids.size(1) - prompt_ids.size(1),
                )

        # ── Step 10: Return dict ────────────────────────────────────────
        num_items_in_batch = completion_mask.sum()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }


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
    base_model_path: str = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    fast_inference: bool = False,
    adapter_model_path: str | None = None,
    mode: str = "train",
    lora_rank: int = 16,
    lora_target_modules: list[str] | None = None,
    lora_dropout: float = 0.0,
    env_name: str = "local",
    gpu_memory_utilization: float | None = None,
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
        base_model_path: Name or path of the original base model.
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
    if env_name == "kaggle" and base_model_path == "unsloth/Qwen3-4B-Instruct-2507":
        base_model_path = "/kaggle/input/models/dzung271828/unsloth/transformers/default/4/qwen3-4b-instruct-2507/qwen3-4b-instruct-2507"

    # ── 0.5 Adapter guard ─────────────────────────────────────────────
    if os.path.isdir(base_model_path) and os.path.exists(
        os.path.join(base_model_path, "adapter_config.json")
    ):
        raise ValueError(
            f"The path '{base_model_path}' appears to be a LoRA adapter "
            "(contains adapter_config.json), not a base model. "
            "Set `adapter_model_path=` to this path and `base_model_path=` "
            "to the original base model (e.g. 'unsloth/Qwen3-4B-Instruct-2507')."
        )

    # ── 0.6 Log resolved paths ─────────────────────────────────────────
    if adapter_model_path is not None:
        print(f"[load_model] base_model_path={base_model_path}, adapter_model_path={adapter_model_path}")
    else:
        print(f"[load_model] base_model_path={base_model_path}")

    # ── 1. Load tokenizer from checkpoint or base model ───────────────
    if adapter_model_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # ── 2. Load the base model (fast_inference handles internal GRPO patching) ─
    kwargs = dict(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        dtype=None,
        gpu_memory_utilization=gpu_memory_utilization if gpu_memory_utilization is not None else (0.3 if os.environ.get("UNSLOTH_VLLM_STANDBY", "0") != "1" else 0.8),
    )
    if mode == "train":
        kwargs["max_lora_rank"] = lora_rank

    model, _ = FastLanguageModel.from_pretrained(**kwargs)

    # ── 4. Load adapter, base model only, or create fresh LoRA ────────
    if adapter_model_path is not None:
        # Checkpoint resume: create Unsloth-patched PeftModel first (so load_lora exists),
        # then load trained adapter weights onto it
        model.resize_token_embeddings(len(tokenizer))
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
        model.load_adapter(adapter_model_path, adapter_name="default")
    elif mode == "inference":
        # Base model only (no adapter) — for benchmarking the base model
        pass
    else:
        # Fresh training: register tokens FIRST, then create LoRA from scratch
        # (Unsloth best practice: add_new_tokens must come before get_peft_model)
        patch_tokenizer_for_custom_roles(tokenizer)
        from unsloth import add_new_tokens
        add_new_tokens(model, tokenizer, REWARD_TOKENS)
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
    train_cfg = config["training"]
    grpo_cfg = config["grpo"]
    data_cfg = config["data"]

    vllm_params = SamplingParams(
        temperature=grpo_cfg["temperature"],
        top_p=0.95,
        min_p=0.05,
        seed=train_cfg["seed"],
        stop=["</tool_call>"],
        include_stop_str_in_output=True,
    )

    return GRPOConfig(
        output_dir=output_dir or train_cfg["output_dir"],
        learning_rate=train_cfg["learning_rate"],  # Table 10: 1e-6
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_generations=train_cfg["num_generations"],  # Table 10: G=5
        max_prompt_length=data_cfg["max_prompt_length"],
        max_completion_length=data_cfg["max_completion_length"],
        temperature=grpo_cfg["temperature"],
        epsilon=grpo_cfg["epsilon"],
        epsilon_high=grpo_cfg.get("epsilon_high", 0.28),  # asymmetric clip (CISPO/DAPO/GT)
        beta=grpo_cfg["kl_coefficient"],  # Table 10: KL coeff beta=0.1
        loss_type=grpo_cfg["loss_type"],
        mask_truncated_completions=grpo_cfg["mask_truncated_completions"],
        vllm_sampling_params=vllm_params,
        # ── vLLM KV cache reduction ────────────────────────────────────
        # Cap KV cache to exactly prompt+completion length rather than
        # full max_seq_length (8192), saving VRAM for training activations.
        save_total_limit=1,
        max_steps=train_cfg["max_steps"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        report_to=train_cfg["report_to"],
        seed=train_cfg["seed"],
        bf16=torch.cuda.is_bf16_supported(),
    )


# ─────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────

def execution_reward(
    completions: list[str],
    ground_truth: list,
    function_library: dict,
    **kwargs,
) -> list[float]:
    """Execution-based R_state reward using the Sandbox, continuous in [0, 1].

    For each completion, creates a fresh ``Sandbox(function_library)``, runs all
    extracted tool calls through it, and returns the fraction of calls that
    succeeded (validated schema + mock execution).  This closes the "syntax but
    non-executable call" hack vector — the agent must produce calls that not only
    *look* correct but also pass schema validation and refer to real functions.

    Args:
        completions: List of model response strings.
        ground_truth: List of ground-truth dicts (or JSON strings; used only for
                      shape alignment, not for scoring).
        function_library: Dict mapping function names to schemas, passed to each
                          fresh ``Sandbox`` instance.
        **kwargs: Additional keyword arguments (ignored; for TRL compatibility).

    Returns:
        List of float scores in [0.0, 1.0].  1.0 if every extracted tool call
        executed successfully.  0.0 if none succeeded.
    """
    rewards = []
    for c, _ in zip(completions, ground_truth):
        calls = extract_all_calls(c)
        if not calls:
            rewards.append(0.0)
            continue
        sandbox = Sandbox(function_library)
        results = sandbox.execute_all(c)
        if not results:
            rewards.append(0.0)
        else:
            rewards.append(sum(1 for r in results if r) / len(results))
    return rewards


def build_trl_reward_functions(
    algorithm: str = "rc_grpo",
    function_library: dict | None = None,
):
    """
    Returns the list of reward functions compatible with TRL's GRPOTrainer
    for the selected algorithm.

    Each function signature: fn(completions: list[str], **kwargs) -> list[float]
    kwargs includes 'ground_truth' passed through from the dataset columns.
    """
    if algorithm == "rc_grpo":
        # RC-GRPO uses the SAME reward functions as vanilla GRPO.
        # The difference is only in prompt conditioning (reward tokens injected
        # by RCGRPOTrainer._generate_and_score_completions) — R(tau) checks
        # state/action correctness identically (see Eq. 5 / Sec 3.1).
        #
        # execution_reward is only added when function_library is provided
        # (it requires a Sandbox and a real function schema source).
        if function_library is not None:
            from functools import partial
            return [
                function_reward,
                argument_reward,
                format_reward,
                partial(execution_reward, function_library=function_library),
            ]
        return [function_reward, argument_reward, format_reward]
    # default / vanilla GRPO
    if function_library is not None:
        from functools import partial
        return [
            function_reward,
            argument_reward,
            format_reward,
            partial(execution_reward, function_library=function_library),
        ]
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

def evaluate_model(
    adapter_model_path: str,
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
    use_vllm: bool = True,
    batch_size: int = 32,
    gpu_memory_utilization: float | None = None,
    base_model_path: str | None = None,
) -> dict:
    """
    Unified evaluation: batched vLLM when use_vllm=True (default, ~10x faster),
    serial HF generate when use_vllm=False.

    gpu_memory_utilization defaults to TRAIN_CONFIG["model"]["gpu_memory_utilization"]
    when not explicitly passed.
    """
    logger = get_logger(__name__)
    tag = "vLLM-Bench" if use_vllm else "Benchmark"
    logger.info(f"[{tag}] Loading model from {adapter_model_path}")

    # Resolve gpu_memory_utilization: explicit arg > config default > 0.8 fallback
    if gpu_memory_utilization is None:
        gpu_memory_utilization = TRAIN_CONFIG["model"]["gpu_memory_utilization"]

    # Resolve base_model_path: explicit arg > TRAIN_CONFIG default
    if base_model_path is None:
        base_model_path = TRAIN_CONFIG["model"]["base_model_path"]

    model, tokenizer = load_model(
        base_model_path=base_model_path,
        adapter_model_path=adapter_model_path,
        mode="inference",
        fast_inference=True,
        gpu_memory_utilization=gpu_memory_utilization,
        env_name=ENV_NAME,
    )

    val_retriever = None
    if argument_values is not None:
        val_retriever = ArgumentValueRetriever(argument_values)

    test_samples = []
    with jsonlines.open(test_dataset_path) as reader:
        for obj in reader:
            test_samples.append(obj)

    suffix = f" (batch_size={batch_size})" if use_vllm else ""
    logger.info(f"[{tag}] {model_name_tag}: evaluating {len(test_samples)} samples{suffix}")

    if use_vllm:
        # ── Batched generation with vLLM ──────────────────────────────
        from vllm import SamplingParams

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

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            stop=["</tool_call>"],
            include_stop_str_in_output=True,
            seed=3407,
        )

        t_start = time.perf_counter()
        outputs = model.fast_generate(prompts, sampling_params)
        gen_time = time.perf_counter() - t_start
        logger.info(f"[{tag}] Batch generation completed in {gen_time:.1f}s ({gen_time/len(prompts):.3f}s/sample)")

        results = []
        for idx, (output, meta) in enumerate(zip(outputs, sample_metas)):
            sample = meta["sample"]
            gt = meta["gt"]
            response = output.outputs[0].text
            latency = (gen_time / len(prompts)) * 1000.0
            cost = estimate_cost(prompts[idx], response)
            metrics = compute_all_metrics(response, gt, sandbox, latency, cost, function_library)
            metrics["sample_id"] = sample.get("id", "")
            results.append(metrics)

    else:
        # ── Serial generation with HF generate ─────────────────────────
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

            # Condition on <|high_reward|> at inference (Sec. 4.1), only
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
    logger.info(f"[{tag}] {model_name_tag} aggregate: {agg}")
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
MODE = "rc_grpo"  # one of: "sft", "grpo", "rc_grpo", "rctp_ft"
assert MODE in ("sft", "grpo", "rc_grpo", "rctp_ft"), \
    f"Unknown MODE: {MODE}. Choose from: sft, grpo, rc_grpo, rctp_ft"
# loss_type is selected via TRAIN_CONFIG["grpo"]["loss_type"] (only used when MODE="rc_grpo")
EVAL_USE_VLLM = True  # Use vLLM batch eval for ~10x speedup
os.environ["KL_COEFFICIENT"] = "0"
DATA_DIR = Path(DATA_MOUNT)
# Kaggle: /kaggle/input/ is read-only — skip mkdir, data is pre-mounted
if ENV_NAME != "kaggle":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

FUNCTION_LIBRARY_PATH = DATA_DIR / "function_library.json"
ARGUMENT_VALUES_PATH = DATA_DIR / "argument_values.json"  # optional

TRAIN_CONFIG = {
    "model": {
        "base_model_path": "unsloth/Qwen3-4B-Instruct-2507",
        "max_seq_length": 8192,
        "load_in_4bit": False,
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
        "output_dir": f"outputs/{MODE}_model{'_' + os.environ.get('KL_COEFFICIENT', 'default') if os.environ.get('KL_COEFFICIENT') else ''}",
        # Table 10 (RC-GRPO RL stage hyperparameters)
        "learning_rate": 5e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.001,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_generations": 16,        # 4-way comparison: G=16
        "max_steps": 500,             # reduce for quick test; increase for full run
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
        "grpo_path": str(DATA_DIR / "grpo_dataset_stage2.jsonl"),
        "rcgrpo_path": str(DATA_DIR / "rcgrpo_dataset_stage2.jsonl"),
        "rctp_path": str(DATA_DIR / "rctp_dataset.jsonl"),
        "max_prompt_length": 7680,
        "max_completion_length": 512,
        "include_all_threshold": 5,
        "eval_batch_size": 128,
    },
    "grpo": {
        "temperature": 1.0,
        "epsilon": 0.2,            # Table 10: symmetric Clip Ratio = 0.2
        "kl_coefficient": float(os.environ.get("KL_COEFFICIENT", "0.0")),
                                   # Override via env var (default 0.0 for
                                   # reference-free comparison across all
                                   # 4 loss types).  Also changeable at
                                   # runtime via TRAIN_CONFIG.
        "loss_type": "grpo",     # Options: "grpo", "dapo", "cispo", "gtpo", "mmr_grpo", "avspo"
        "epsilon_high": 0.28,     # Upper clip bound (CISPO/DAPO asymmetric clip)
        "mask_truncated_completions": True,
    },
    "gtpo": {
        "entropy_threshold": 0.5,  # δ filter: skip trajectories with mean entropy > this
        "gamma": 0.01,              # γ: entropy regularization coefficient
    },
    "mmr": {
        "enabled": False,          # Master toggle for MMR diversity reweighting
        "lambda_div": 0.7,         # λ: diversity weight (only used in 'original' mode)
        "mode": "sigmoid",         # One of: "original", "std", "sigmoid"
        "mmr_std_temp": 2.0,       # Temperature for std-based adaptive λ (modes "std"/"sigmoid")
    },
    "avspo": {
        "acr_threshold": 1e-6,     # τ: collapse detection threshold (Definition 4.1)
        "alpha": 0.5,              # α: sensitivity for K = max(1, min(G, ceil(G·ACR^α)))
        "anchor_reward": 0.1,      # r_anchor: virtual reward scale for all-incorrect groups
        "tau_adapt_initial": 0.5,  # τ_adapt^(0): initial adaptive threshold
        "tau_adapt_eta": 0.01,     # η: adaptive threshold learning rate
        "acr_log_interval": 10,    # Logging interval for ACR diagnostics
    },
    "sft": {
        "output_dir": "outputs/sft_model",
        "learning_rate": 2e-5,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_epochs": 2,
        "max_seq_length": 8192,
    },
    # RCTP-FT (Stage 1) hyperparameters — Table 11
    "rctp_ft": {
        "output_dir": "outputs/rctp_ft_model",
        "learning_rate": 2e-5,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_epochs": 2,
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
```

```python
# ── Smart truncation safety net ─────────────────────────────────────
# smart_truncate was removed — it was legacy (disabled) code.
# See AGENTS.md: "max_seq_length=8192 fits all samples"
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
    model, tokenizer = load_model(
        base_model_path=TRAIN_CONFIG["model"]["base_model_path"],
        max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],
        load_in_4bit=TRAIN_CONFIG["model"]["load_in_4bit"],
        fast_inference=False,
        adapter_model_path=None,
        mode="train",
        lora_rank=TRAIN_CONFIG["lora"]["r"],
        lora_target_modules=TRAIN_CONFIG["lora"]["target_modules"],
        lora_dropout=TRAIN_CONFIG["lora"]["lora_dropout"],
        env_name=ENV_NAME,
    )
    print("Model and tokenizer loaded.")

    print("Tokenizer special tokens:")
    print(f"  bos_token: {tokenizer.bos_token}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  additional_special_tokens: {tokenizer.additional_special_tokens}")

    sft_cfg = TRAIN_CONFIG["sft"]
    raw_records = load_jsonl(TRAIN_CONFIG["data"]["sft_path"])
    from datasets import Dataset

    # Use full text field directly; train_on_responses_only will mask
    # everything except the assistant response.
    dataset = Dataset.from_list(raw_records)

    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=sft_cfg["output_dir"],
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
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
        print(f"  {k}: {str(v)}")
    print("===========================")

    # ── Hyperparameter snapshot ──
    _sft_table = [
        ["Mode", "SFT - Supervised Fine-Tuning"],
        ["Dataset", TRAIN_CONFIG["data"]["sft_path"]],
        ["Samples", len(dataset)],
        ["Base model", TRAIN_CONFIG["model"]["base_model_path"]],
        ["Max seq length", TRAIN_CONFIG["model"]["max_seq_length"]],
        ["Load in 4bit", TRAIN_CONFIG["model"]["load_in_4bit"]],
        ["Full finetuning", TRAIN_CONFIG["model"]["full_finetuning"]],
        ["LoRA rank (r)", TRAIN_CONFIG["lora"]["r"]],
        ["LoRA dropout", TRAIN_CONFIG["lora"]["lora_dropout"]],
        ["LoRA targets", TRAIN_CONFIG["lora"]["target_modules"]],
        ["LoRA bias", TRAIN_CONFIG["lora"]["bias"]],
        ["Gradient checkpointing", TRAIN_CONFIG["lora"]["use_gradient_checkpointing"]],
        ["Learning rate", training_args.learning_rate],
        ["Batch size (per device)", training_args.per_device_train_batch_size],
        ["Gradient accumulation steps", training_args.gradient_accumulation_steps],
        ["Num epochs", training_args.num_train_epochs],
        ["Max grad norm", training_args.max_grad_norm],
        ["Weight decay", training_args.weight_decay],
        ["Optimizer", training_args.optim],
        ["Warmup ratio", training_args.warmup_ratio],
        ["Seed", training_args.seed],
        ["Logging steps", training_args.logging_steps],
        ["Save strategy", training_args.save_strategy],
        ["Save total limit", training_args.save_total_limit],
        ["Dataloader workers", training_args.dataloader_num_workers],
        ["FP16", training_args.fp16],
        ["BF16", training_args.bf16],
        ["Precision", "bf16" if torch.cuda.is_bf16_supported() else "fp16"],
    ]
    print(tabulate(_sft_table, headers=["Hyperparameter", "Value"], tablefmt="fancy_grid"))

    trainer.train()
    trainer.save_model(sft_cfg["output_dir"])
    tokenizer.save_pretrained(Path(sft_cfg["output_dir"]))

    # ── Save hyperparameters ──
    _sft_hp = {
        "mode": "sft",
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": {
            "path": TRAIN_CONFIG["data"]["sft_path"],
            "samples": len(dataset),
        },
        "model": {
            "base_model": TRAIN_CONFIG["model"]["base_model_path"],
            "max_seq_length": TRAIN_CONFIG["model"]["max_seq_length"],
            "load_in_4bit": TRAIN_CONFIG["model"]["load_in_4bit"],
            "full_finetuning": TRAIN_CONFIG["model"]["full_finetuning"],
        },
        "lora": {
            "r": TRAIN_CONFIG["lora"]["r"],
            "lora_dropout": TRAIN_CONFIG["lora"]["lora_dropout"],
            "target_modules": TRAIN_CONFIG["lora"]["target_modules"],
            "bias": TRAIN_CONFIG["lora"]["bias"],
            "use_gradient_checkpointing": TRAIN_CONFIG["lora"]["use_gradient_checkpointing"],
        },
        "training": {
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "num_train_epochs": training_args.num_train_epochs,
            "max_grad_norm": training_args.max_grad_norm,
            "weight_decay": training_args.weight_decay,
            "optim": training_args.optim,
            "warmup_ratio": training_args.warmup_ratio,
            "seed": training_args.seed,
            "logging_steps": training_args.logging_steps,
            "save_strategy": training_args.save_strategy,
            "save_total_limit": training_args.save_total_limit,
            "dataloader_num_workers": training_args.dataloader_num_workers,
        },
        "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
    }
    _sft_out = Path(sft_cfg["output_dir"])
    _sft_out.mkdir(parents=True, exist_ok=True)
    with open(_sft_out / "training_hyperparameters.json", "w") as f:
        json.dump(_sft_hp, f, indent=2, ensure_ascii=False)

    print(f"[SFT] Model saved -> {sft_cfg['output_dir']}")

    # ── Free training resources for clean evaluation ──
    cleanup_training(
        trainer=trainer if 'trainer' in dir() else None,
        model=model if 'model' in dir() else None,
        tokenizer=tokenizer if 'tokenizer' in dir() else None,
    )
```

```python
if MODE == "sft":
    !zip -r sft_model.zip outputs/sft_model
    from IPython.display import FileLink
    FileLink("sft_model.zip")
```

```python
# ===================== STAGE 1: RCTP-FT (only for rc_grpo) =====================
SAMPLE_LOG = get_logger("sample")
high_reward_probability = 0.5  # overwritten below from real data stats

if MODE == "rctp_ft":
    print("\n" + "=" * 70)
    print("STAGE 1: Reward-Conditioned Trajectory Policy (RCTP) Fine-tuning")
    print("=" * 70)
    model, tokenizer = load_model(
        base_model_path=TRAIN_CONFIG["model"]["base_model_path"],
        max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],
        load_in_4bit=TRAIN_CONFIG["model"]["load_in_4bit"],
        fast_inference=False,
        adapter_model_path=None,
        mode="train",
        lora_rank=TRAIN_CONFIG["lora"]["r"],
        lora_target_modules=TRAIN_CONFIG["lora"]["target_modules"],
        lora_dropout=TRAIN_CONFIG["lora"]["lora_dropout"],
        env_name=ENV_NAME,
    )
    print("Model and tokenizer loaded.")

    print("Tokenizer special tokens:")
    print(f"  bos_token: {tokenizer.bos_token}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  additional_special_tokens: {tokenizer.additional_special_tokens}")

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
    train_dataset = Dataset.from_list([{"text": t} for t in formatted_texts])

    training_args = SFTConfig(
        output_dir=rctp_cfg["output_dir"],
        per_device_train_batch_size=rctp_cfg["batch_size"],
        gradient_accumulation_steps=rctp_cfg["gradient_accumulation_steps"],
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
        save_total_limit=1,
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
        print(f"  {k}: {str(v)}")
    print("===========================")

    SAMPLE_LOG.info("stage=pre_stage1_train type=summary trajectories=%d epochs=%d batch_size=%d",
                    len(rctp_trajectories), training_args.num_train_epochs, training_args.per_device_train_batch_size)

    # ── Hyperparameter snapshot ──
    _rctp_table = [
        ["Mode", "RCTP-FT (Stage 1)"],
        ["Dataset", TRAIN_CONFIG["data"]["rctp_path"]],
        ["Trajectories", len(rctp_trajectories)],
        ["High reward probability", high_reward_probability],
        ["Base model", TRAIN_CONFIG["model"]["base_model_path"]],
        ["Max seq length", TRAIN_CONFIG["model"]["max_seq_length"]],
        ["Load in 4bit", TRAIN_CONFIG["model"]["load_in_4bit"]],
        ["LoRA rank (r)", TRAIN_CONFIG["lora"]["r"]],
        ["LoRA dropout", TRAIN_CONFIG["lora"]["lora_dropout"]],
        ["LoRA targets", TRAIN_CONFIG["lora"]["target_modules"]],
        ["LoRA bias", TRAIN_CONFIG["lora"]["bias"]],
        ["Gradient checkpointing", TRAIN_CONFIG["lora"]["use_gradient_checkpointing"]],
        ["Learning rate", training_args.learning_rate],
        ["Batch size (per device)", training_args.per_device_train_batch_size],
        ["Gradient accumulation steps", training_args.gradient_accumulation_steps],
        ["Num epochs", training_args.num_train_epochs],
        ["Max grad norm", training_args.max_grad_norm],
        ["Weight decay", training_args.weight_decay],
        ["Optimizer", training_args.optim],
        ["Warmup ratio", training_args.warmup_ratio],
        ["Seed", training_args.seed],
        ["Logging steps", training_args.logging_steps],
        ["Save strategy", training_args.save_strategy],
        ["Save total limit", training_args.save_total_limit],
        ["Dataloader workers", training_args.dataloader_num_workers],
        ["FP16", training_args.fp16],
        ["BF16", training_args.bf16],
        ["Precision", "bf16" if torch.cuda.is_bf16_supported() else "fp16"],
    ]
    print(tabulate(_rctp_table, headers=["Hyperparameter", "Value"], tablefmt="fancy_grid"))

    trainer.train()
    trainer.save_model(rctp_cfg["output_dir"])
    tokenizer.save_pretrained(Path(rctp_cfg["output_dir"]))

    # ── Save hyperparameters ──
    _rctp_hp = {
        "mode": "rctp_ft",
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": {
            "path": TRAIN_CONFIG["data"]["rctp_path"],
            "trajectories": len(rctp_trajectories),
            "high_reward_probability": high_reward_probability,
        },
        "model": {
            "base_model": TRAIN_CONFIG["model"]["base_model_path"],
            "max_seq_length": TRAIN_CONFIG["model"]["max_seq_length"],
            "load_in_4bit": TRAIN_CONFIG["model"]["load_in_4bit"],
        },
        "lora": {
            "r": TRAIN_CONFIG["lora"]["r"],
            "lora_dropout": TRAIN_CONFIG["lora"]["lora_dropout"],
            "target_modules": TRAIN_CONFIG["lora"]["target_modules"],
            "bias": TRAIN_CONFIG["lora"]["bias"],
            "use_gradient_checkpointing": TRAIN_CONFIG["lora"]["use_gradient_checkpointing"],
        },
        "training": {
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "num_train_epochs": training_args.num_train_epochs,
            "max_grad_norm": training_args.max_grad_norm,
            "weight_decay": training_args.weight_decay,
            "optim": training_args.optim,
            "warmup_ratio": training_args.warmup_ratio,
            "seed": training_args.seed,
            "logging_steps": training_args.logging_steps,
            "save_strategy": training_args.save_strategy,
            "save_total_limit": training_args.save_total_limit,
            "dataloader_num_workers": training_args.dataloader_num_workers,
        },
        "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
    }
    _rctp_out = Path(rctp_cfg["output_dir"])
    _rctp_out.mkdir(parents=True, exist_ok=True)
    with open(_rctp_out / "training_hyperparameters.json", "w") as f:
        json.dump(_rctp_hp, f, indent=2, ensure_ascii=False)

    print(f"[Stage 1] RCTP-FT checkpoint saved -> {rctp_cfg['output_dir']}")
    print("(This checkpoint becomes pi_ref for Stage 2, per Algorithm 1 line 9.)")

    # ── Free training resources for clean evaluation ──
    cleanup_training(
        trainer=trainer if 'trainer' in dir() else None,
        model=model if 'model' in dir() else None,
        tokenizer=tokenizer if 'tokenizer' in dir() else None,
    )
```

```python
if MODE == "rctp_ft":
    !zip -r rctp_ft_model.zip outputs/rctp_ft_model
    from IPython.display import FileLink
    FileLink("rctp_ft_model.zip")
```

```python
# ===================== Unified model loader for Stage 2 =====================
# Loads a base model with a fine-tuned LoRA adapter for inference or training,
# controlled by the `mode` argument.  Named parameters base_model_path /
# adapter_model_path make it clear what each path refers to.
# (Now handled by the unified load_model() above.)

```

```python
if MODE in ("grpo", "rc_grpo"):
    # ===================== STAGE 2: RC-GRPO (or vanilla GRPO baseline) =====================
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"  # required by vLLM standby mode 
    os.environ["UNSLOTH_VLLM_STANDBY"] = "0"  # vLLM standby mode — shares GPU with trainer
    print("\n" + "=" * 70)
    print(f"STAGE 2: {'RC-GRPO' if MODE == 'rc_grpo' else 'Vanilla GRPO baseline'}")
    print("=" * 70)

    # ── Set model paths for Stage 2 ──────────────────────────────────
    # The adapter checkpoint is typically the Stage 1 (RCTP-FT) output.
    # Adjust these to point to your trained checkpoint directories.
    base_model_path = "unsloth/Qwen3-4B-Instruct-2507"
    adapter_model_path = "/kaggle/input/models/dzung271828/unsloth/transformers/default/4/rctp_ft_model/rctp_ft_model"  # Stage 1 (RCTP-FT) checkpoint

    print(f"[Stage 2] Loading base_model_path = {base_model_path}")
    print(f"[Stage 2] Loading adapter_model_path = {adapter_model_path}")

    model, tokenizer = load_model(
        base_model_path=base_model_path,
        adapter_model_path=adapter_model_path,
        mode="train",          # keep in train mode for further fine-tuning
        fast_inference=True,
        max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],
        load_in_4bit=TRAIN_CONFIG["model"]["load_in_4bit"],
        env_name=ENV_NAME,
    )
    print("Model and tokenizer loaded.")
    print("Tokenizer special tokens:")
    print(f"  bos_token: {tokenizer.bos_token}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  additional_special_tokens: {tokenizer.additional_special_tokens}")

    from datasets import Dataset
    if MODE == "rc_grpo":
        data_path = TRAIN_CONFIG["data"].get("rcgrpo_path", TRAIN_CONFIG["data"]["grpo_path"])
    else:
        data_path = TRAIN_CONFIG["data"]["grpo_path"]
    dataset = Dataset.from_list(load_jsonl(data_path))

    grpo_args = build_grpo_config(TRAIN_CONFIG, output_dir=TRAIN_CONFIG["training"]["output_dir"])

    if MODE == "rc_grpo":
        loss_type = TRAIN_CONFIG["grpo"]["loss_type"]
        shared_kwargs = dict(
            model=model,
            processing_class=tokenizer,
            reward_funcs=build_trl_reward_functions("rc_grpo"),
            args=grpo_args,
            train_dataset=dataset,
            high_reward_probability=high_reward_probability,
        )
        if loss_type == "gtpo":
            trainer = GTPOTrainer(
                **shared_kwargs,
                entropy_threshold=TRAIN_CONFIG["gtpo"]["entropy_threshold"],
                gamma=TRAIN_CONFIG["gtpo"]["gamma"],
            )
        elif loss_type == "mmr_grpo":
            trainer = MMRGRPOTrainer(
                **shared_kwargs,
                mmr_config=TRAIN_CONFIG["mmr"],
            )
        elif loss_type == "avspo":
            trainer = AVSPOTrainer(
                **shared_kwargs,
                avspo_config=TRAIN_CONFIG.get("avspo", {}),
            )
        else:
            trainer = RCGRPOTrainer(**shared_kwargs)
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
        print(f"  {k}: {str(v)}")
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

    # ── Hyperparameter snapshot ──
    _grpo_table = [
        ["Algorithm", f"RC-{TRAIN_CONFIG['grpo']['loss_type'].upper()}" if MODE == "rc_grpo" else "Vanilla GRPO"],
        ["Trainer", trainer.__class__.__name__],
        ["Dataset", data_path],
        ["Samples", len(dataset)],
        ["Base model", base_model_path],
        ["Adapter model", adapter_model_path],
        ["Base model name", TRAIN_CONFIG["model"]["base_model_path"]],
        ["Max seq length", TRAIN_CONFIG["model"]["max_seq_length"]],
        ["Load in 4bit", TRAIN_CONFIG["model"]["load_in_4bit"]],
        ["GPU memory util (Unsloth)", TRAIN_CONFIG["model"]["gpu_memory_utilization"]],
        ["Full finetuning", TRAIN_CONFIG["model"]["full_finetuning"]],
        ["LoRA rank (r)", TRAIN_CONFIG["lora"]["r"]],
        ["LoRA dropout", TRAIN_CONFIG["lora"]["lora_dropout"]],
        ["LoRA targets", TRAIN_CONFIG["lora"]["target_modules"]],
        ["LoRA bias", TRAIN_CONFIG["lora"]["bias"]],
        ["Learning rate", TRAIN_CONFIG["training"]["learning_rate"]],
        ["Adam beta1", TRAIN_CONFIG["training"]["adam_beta1"]],
        ["Adam beta2", TRAIN_CONFIG["training"]["adam_beta2"]],
        ["Weight decay", TRAIN_CONFIG["training"]["weight_decay"]],
        ["Warmup ratio", TRAIN_CONFIG["training"]["warmup_ratio"]],
        ["LR scheduler", TRAIN_CONFIG["training"]["lr_scheduler_type"]],
        ["Optimizer", TRAIN_CONFIG["training"]["optim"]],
        ["Batch size (per device)", TRAIN_CONFIG["training"]["per_device_train_batch_size"]],
        ["Gradient accumulation steps", TRAIN_CONFIG["training"]["gradient_accumulation_steps"]],
        ["Num generations", TRAIN_CONFIG["training"]["num_generations"]],
        ["Max steps", TRAIN_CONFIG["training"]["max_steps"]],
        ["Save steps", TRAIN_CONFIG["training"]["save_steps"]],
        ["Logging steps", TRAIN_CONFIG["training"]["logging_steps"]],
        ["Max grad norm", TRAIN_CONFIG["training"]["max_grad_norm"]],
        ["Seed", TRAIN_CONFIG["training"]["seed"]],
        ["Temperature", TRAIN_CONFIG["grpo"]["temperature"]],
        ["Epsilon (clip)", TRAIN_CONFIG["grpo"]["epsilon"]],
        ["Epsilon high", TRAIN_CONFIG["grpo"].get("epsilon_high", "N/A")],
        ["KL coefficient (beta)", TRAIN_CONFIG["grpo"]["kl_coefficient"]],
        ["Loss type", TRAIN_CONFIG["grpo"]["loss_type"]],
        ["MMR enabled", TRAIN_CONFIG["mmr"]["enabled"]],
        ["MMR mode", TRAIN_CONFIG["mmr"]["mode"]],
        ["MMR lambda", TRAIN_CONFIG["mmr"]["lambda_div"]],
        ["ACR threshold", TRAIN_CONFIG.get("avspo", {}).get("acr_threshold", "N/A")],
        ["ACR alpha", TRAIN_CONFIG.get("avspo", {}).get("alpha", "N/A")],
        ["Anchor reward", TRAIN_CONFIG.get("avspo", {}).get("anchor_reward", "N/A")],
        ["Tau adapt initial", TRAIN_CONFIG.get("avspo", {}).get("tau_adapt_initial", "N/A")],
        ["Tau adapt eta", TRAIN_CONFIG.get("avspo", {}).get("tau_adapt_eta", "N/A")],
        ["ACR log interval", TRAIN_CONFIG.get("avspo", {}).get("acr_log_interval", "N/A")],
        ["Mask truncated", TRAIN_CONFIG["grpo"]["mask_truncated_completions"]],
        ["Max prompt length", TRAIN_CONFIG["data"]["max_prompt_length"]],
        ["Max completion length", TRAIN_CONFIG["data"]["max_completion_length"]],
        ["Precision", "bf16" if torch.cuda.is_bf16_supported() else "fp16"],
    ]
    print(tabulate(_grpo_table, headers=["Hyperparameter", "Value"], tablefmt="fancy_grid"))

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # ── Auto-save model and tokenizer after training ──
    output_dir = MODE_OUTPUT_DIR
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Save hyperparameters ──
    _grpo_hp = {
        "mode": MODE,
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": {
            "path": data_path,
            "samples": len(dataset),
        },
        "model": {
            "base_model": TRAIN_CONFIG["model"]["base_model_path"],
            "base_model_path": base_model_path,
            "adapter_model_path": adapter_model_path,
            "max_seq_length": TRAIN_CONFIG["model"]["max_seq_length"],
            "load_in_4bit": TRAIN_CONFIG["model"]["load_in_4bit"],
            "gpu_memory_utilization": TRAIN_CONFIG["model"]["gpu_memory_utilization"],
            "full_finetuning": TRAIN_CONFIG["model"]["full_finetuning"],
        },
        "lora": {
            "r": TRAIN_CONFIG["lora"]["r"],
            "lora_dropout": TRAIN_CONFIG["lora"]["lora_dropout"],
            "target_modules": TRAIN_CONFIG["lora"]["target_modules"],
            "bias": TRAIN_CONFIG["lora"]["bias"],
        },
        "training": {
            "learning_rate": TRAIN_CONFIG["training"]["learning_rate"],
            "adam_beta1": TRAIN_CONFIG["training"]["adam_beta1"],
            "adam_beta2": TRAIN_CONFIG["training"]["adam_beta2"],
            "weight_decay": TRAIN_CONFIG["training"]["weight_decay"],
            "warmup_ratio": TRAIN_CONFIG["training"]["warmup_ratio"],
            "lr_scheduler_type": TRAIN_CONFIG["training"]["lr_scheduler_type"],
            "optim": TRAIN_CONFIG["training"]["optim"],
            "per_device_train_batch_size": TRAIN_CONFIG["training"]["per_device_train_batch_size"],
            "gradient_accumulation_steps": TRAIN_CONFIG["training"]["gradient_accumulation_steps"],
            "num_generations": TRAIN_CONFIG["training"]["num_generations"],
            "max_steps": TRAIN_CONFIG["training"]["max_steps"],
            "save_steps": TRAIN_CONFIG["training"]["save_steps"],
            "logging_steps": TRAIN_CONFIG["training"]["logging_steps"],
            "max_grad_norm": TRAIN_CONFIG["training"]["max_grad_norm"],
            "seed": TRAIN_CONFIG["training"]["seed"],
        },
        "rl": {
            "temperature": TRAIN_CONFIG["grpo"]["temperature"],
            "epsilon": TRAIN_CONFIG["grpo"]["epsilon"],
            "epsilon_high": TRAIN_CONFIG["grpo"].get("epsilon_high"),
            "beta": TRAIN_CONFIG["grpo"]["kl_coefficient"],
            "loss_type": TRAIN_CONFIG["grpo"]["loss_type"],
            "mask_truncated_completions": TRAIN_CONFIG["grpo"]["mask_truncated_completions"],
        },
        "gtpo": dict(TRAIN_CONFIG.get("gtpo", {})),
        "mmr": dict(TRAIN_CONFIG.get("mmr", {})),
        "avspo": dict(TRAIN_CONFIG.get("avspo", {})),
        "data": {
            "max_prompt_length": TRAIN_CONFIG["data"]["max_prompt_length"],
            "max_completion_length": TRAIN_CONFIG["data"]["max_completion_length"],
        },
        "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
    }
    _grpo_out = Path(output_dir)
    _grpo_out.mkdir(parents=True, exist_ok=True)
    with open(_grpo_out / "training_hyperparameters.json", "w") as f:
        json.dump(_grpo_hp, f, indent=2, ensure_ascii=False)

    print(f"Model saved to {output_dir}")
    !zip -r {output_dir}.zip {output_dir}
    from IPython.display import FileLink
    FileLink(f"{output_dir}.zip")
else:
    print(f"Skipping Stage 2 (MODE={MODE}).")
```

```python
# ── Free training resources for clean evaluation ──
# Releases: vLLM engine, distributed state, model, LoRA adapter, CUDA cache.
# Uses the cleanup_training() helper defined in the utilities section above.
cleanup_training(
    trainer=trainer if 'trainer' in dir() else None,
    model=model if 'model' in dir() else None,
    tokenizer=tokenizer if 'tokenizer' in dir() else None,
)
# Reset allocator to clear stale graph pools (applied after cleanup)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
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
        local_dir=f"{BASE_OUTPUT_PATH}/outputs/",
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    print("[Colab] Download complete.")
```

```python
# ===================== EVALUATION =====================
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"  # FIX: avoid stale CUDA graph pool crash (use_count > 0 assertion)
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"  # eval: ensures no dual vLLM engines; gpu_memory_utilization is passed explicitly from TRAIN_CONFIG
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"  # Reset CUDA allocator to clear stale graph pools
# MODE_OUTPUT_DIR was already set from TRAIN_CONFIG["training"]["output_dir"] above — no override needed
test_dataset_path = TRAIN_CONFIG["data"]["test_path"]
if Path(test_dataset_path).exists():
    retriever = FunctionRetriever(function_library, method="hybrid")
    sandbox = Sandbox(function_library)

    eval_result = evaluate_model(
        adapter_model_path=MODE_OUTPUT_DIR,
        test_dataset_path=test_dataset_path,
        function_library=function_library,
        retriever=retriever,
        sandbox=sandbox,
        top_k=5,
        max_new_tokens=512,
        model_name_tag=MODE,
        condition_on_high_reward=(MODE == "rc_grpo"),
        argument_values=argument_values_catalog,
        use_vllm=EVAL_USE_VLLM,
        batch_size=TRAIN_CONFIG["data"]["eval_batch_size"],
        gpu_memory_utilization=TRAIN_CONFIG["model"]["gpu_memory_utilization"],
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
    from IPython.display import FileLink
    FileLink("outputs/evaluation_reports/metrics_summary.csv")
    FileLink("outputs/evaluation_reports/full_results.json")
else:
    print("Test dataset not found; skipping evaluation.")
```

```python
try:
    del model
    del tokenizer
except:
    pass
import gc
gc.collect()
torch.cuda.empty_cache()
```
