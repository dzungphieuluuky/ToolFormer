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

<!-- #region _cell_guid="8618e9e3-3d65-42a3-a06d-71941a7ba3f6" _uuid="12421c11-7b57-4b19-ae4a-c8f151f33f76" jupyter={"outputs_hidden": false} -->
# Nemotron finetuning pipeline
<!-- #endregion -->

```python _cell_guid="3333948f-b85b-4916-90bf-9937b9faeb6f" _uuid="0e774cb7-6e44-4ead-95e9-c1e515b41ac3" jupyter={"outputs_hidden": false}
#@title Environment-Aware Installation
import os
import sys
from IPython import get_ipython

# Detect environment
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
    
# Detect GPU type
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

```python _cell_guid="96e3dd3d-064b-4320-bf45-8f3b683df8ce" _uuid="cdcb4ca9-08df-49d8-b57a-653656b25a86" jupyter={"outputs_hidden": false}
# Base model path (auto-detects Kaggle path)
if ENV_NAME == "kaggle":
    BASE_MODEL_NAME = "/kaggle/input/models/dzung271828/unsloth/transformers/default/6/qwen3-4b-instruct-2507/qwen3-4b-instruct-2507"
else:
    BASE_MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
print(f"Base model: {BASE_MODEL_NAME}")
```

```python _cell_guid="ea0a0654-fff0-4ea2-a6a1-6de10dd27398" _uuid="ed25c213-2316-41d5-8197-1daa820ae42f" jupyter={"outputs_hidden": false}
# Install packages
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

```python _cell_guid="1ae73acc-d4e1-4b99-ad1a-e981b348ed15" _uuid="587bd2ea-cfd3-45e2-8fea-45af12ef667b" jupyter={"outputs_hidden": false}
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

```python _cell_guid="0c074557-9210-40b9-915f-c153b8ab70ae" _uuid="0e756f96-5e62-499d-b35a-154945d9a102" jupyter={"outputs_hidden": false}
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

```python _cell_guid="626c2ec5-0058-47ae-a257-94583f59a658" _uuid="caaa16d8-4278-45a4-af8d-14bb1cb7f3e7" jupyter={"outputs_hidden": false}
import os
import sys
from IPython import get_ipython

def load_secret(key_name: str) -> str | None:
    """Load secret from env-specific store (colab/kaggle/os.environ)."""
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
    """Print Python, PyTorch, CUDA, GPU info."""
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

```python _cell_guid="35e61203-f1a2-4227-9b08-8b8544ed840a" _uuid="8bd83f46-6469-47ac-bb0d-f1be347a651b" jupyter={"outputs_hidden": false}
!find /usr -name "libcuda.so*" 2>/dev/null
```

```python _cell_guid="2898306f-c6a6-4bc8-b503-45f7925d705d" _uuid="82257ded-c78c-44da-bbde-0fa2058e5b9f" jupyter={"outputs_hidden": false}
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

```python _cell_guid="932325f8-6c85-4d21-8e64-620d0a94a284" _uuid="3b969ce3-48e1-400d-aef1-58e589b2487b" jupyter={"outputs_hidden": false}
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

```python _cell_guid="ca5fe2ac-1262-454f-89e0-038add5b0a8e" _uuid="09686b9f-f9e0-4c0e-b61c-0e73342cbb9a" jupyter={"outputs_hidden": false}
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

```python _cell_guid="6d6df671-103e-48cf-8f57-83dc2abeb4c3" _uuid="fea4a0c2-f7ba-4690-baa6-27861d7c5f13" jupyter={"outputs_hidden": false}
# ===================== logging_utils.py =====================
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """RichHandler-based logger."""
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
    """Generate model response for a single prompt."""
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

```python _cell_guid="ee611b23-ff70-441a-9adb-cf80f6bc3722" _uuid="29d80f63-4e98-4eb9-8120-c2d5ee5bf0ba" jupyter={"outputs_hidden": false}
from typing import Any, Callable


def _default_mock(function_name: str, arguments: dict) -> dict:
    return {
        "status": "success",
        "function": function_name,
        "result": f"Mock result for {function_name}({arguments})",
    }


class Sandbox:
    """Executes mock tool calls, tracks cumulative state for R_state comparison."""

    def __init__(
        self,
        function_library: dict,
        mocks: dict[str, Callable] | None = None,
        timeout_seconds: float = 5.0,
    ):
        """fn schema dict, optional mock callables, execution timeout."""
        self.library = function_library
        self.mocks = mocks or {}
        self.timeout = timeout_seconds
        self._call_log: list[dict] = []
        self._state: dict = {}

    # --- public API ---
    def execute(self, call_input: str | dict | None) -> bool:
        """Execute single tool call. Returns success."""
        if call_input is None:
            return False
        call = self._resolve_call(call_input)
        if call is None:
            return False
        return self._run_call(call)

    def execute_all(self, response: str) -> list[bool]:
        """Parse and execute all tool calls from model output."""
        calls = extract_all_calls(response)
        return [self._run_call(c) for c in calls]

    def get_call_log(self) -> list[dict]:
        return self._call_log.copy()

    def get_state(self) -> dict:
        """Cumulative env state snapshot."""
        return self._state.copy()

    def clear(self) -> None:
        self._call_log.clear()
        self._state.clear()

    def clear_log(self) -> None:
        """Alias for clear()."""
        self.clear()

# internals
    def _resolve_call(self, call_input: str | dict) -> dict | None:
        """Parse call input into standard dict. None on failure."""
        if isinstance(call_input, dict):
            return call_input
        return extract_call(call_input)

    def _run_call(self, call: dict) -> bool:
        """Validate schema, run mock, log, track state."""
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
        """Validate required params + constraints (min, max, enum)."""
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
        """Append entry to call log."""
        self._call_log.append({
            "function": func_name,
            "arguments": arguments,
            "status": status,
            "result": result,
        })
```

```python _cell_guid="ea947743-3545-43de-9b5e-117a584f373b" _uuid="e8644c66-9330-4f84-aa15-89d26629a39d" jupyter={"outputs_hidden": false}
# ===================== vietnamese_normalizer.py =====================
"""Vietnamese text normalization for cross-script (diacritics ↔ ASCII ↔ codes)."""

import re
import unicodedata
from functools import lru_cache


# NFKD strips most diacritics but misses đ/Đ

_VN_CHAR_MAP = {
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "ơ": "o", "Ơ": "O",
    "ư": "u", "Ư": "U",
}

_MULTI_SPACE = re.compile(r"\s+")

# Common Vietnamese abbreviations/synonyms
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
    """Lower → NFKD → char-map (đ→d) → strip combining → collapse ws.
    "Hà Nội" → "ha noi", "Đà Nẵng" → "da nang"."""
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
    """Expand Vietnamese abbreviations. "tp hcm" → "thanh pho ho chi minh"."""
    tokens = text_normalized.split()
    expanded = []
    for t in tokens:
        if t in _VN_SYNONYMS:
            expanded.append(_VN_SYNONYMS[t])
        else:
            expanded.append(t)
    return " ".join(expanded)


def tokenize_meaningful(text_normalized: str, min_len: int = 2) -> set[str]:
    """Skip stopwords and tokens < min_len."""
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

```python _cell_guid="0d1e9f10-231f-4b75-92cf-f9ca4fa45bf8" _uuid="91a2f11d-6d13-4dc8-be42-e816d6899046" jupyter={"outputs_hidden": false}
# ===================== value_catalog.py =====================
"""Deterministic alias-based value lookup (no embeddings).
Small enums → all values. Large → alias + token match. Code in query → exact."""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class CatalogEntry:
    """One possible parameter value."""
    code: str
    label: str
    group: str = ""
    alt_label: str = ""
    # Normalized forms built once at load
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
        self._label_tokens = tokenize_meaningful(expand_synonyms(all_text))

    def add_alias(self, alias: str) -> None:
        """Add alias from hand-curated alias table."""
        norm = normalize_vietnamese(alias)
        self._aliases.append(norm)
        self._label_tokens = self._label_tokens | tokenize_meaningful(norm)


@dataclass
class ValueMatch:
    """Value matching result."""
    code: str
    label: str
    group: str
    score: float = 0.0
    alt_label: str = ""


class ValueCatalog:
    """
    Argument value lookup for one parameter type.
    Priority: exact code → label → alias → token overlap → n-gram (0.2–1.0).
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

        # Reverse alias map: normalized_alias → entry index
        self._alias_to_idx: dict[str, int] = {}
        for i, entry in enumerate(entries):
            self._alias_to_idx[entry._norm_label] = i
            if entry._norm_alt:
                self._alias_to_idx[entry._norm_alt] = i
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
        """Score all entries, return top-k."""
        scored: list[tuple[float, CatalogEntry]] = []

        for entry in self.entries:
            score = self._score_entry(
                entry, query_normalized, query_tokens
            )
            if score > 0.0:
                scored.append((score, entry))

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
        """All values (for small enums)."""
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
        # P1: Exact code in query
        code_lower = entry.code.lower()
        if len(code_lower) >= 2 and code_lower in query_norm:
            specificity = min(1.0, len(entry.code) / 5.0)
            return 0.85 + 0.15 * specificity

        # P2: Full label substring
        if entry._norm_label and entry._norm_label in query_norm:
            return 0.95

        # P3: Alt label substring
        if entry._norm_alt and entry._norm_alt in query_norm:
            return 0.90

        # P4: Alias match
        for alias in entry._aliases:
            if alias in query_norm:
                return 0.88

        # P5: Alias table reverse lookup
        for alias_str, idx in self._alias_to_idx.items():
            if self.entries[idx] is entry and len(alias_str) >= 3:
                if alias_str in query_norm:
                    return 0.85

        # P6: Token overlap
        if entry._label_tokens and query_tokens:
            overlap = query_tokens & entry._label_tokens
            if overlap:
                coverage = len(overlap) / len(entry._label_tokens)
                return min(0.70, 0.25 + 0.45 * coverage)

        # P7: Bigram similarity (short codes)
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
    """Load argument value catalogs from JSON. Optional aliases file for manual overrides."""
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

```python _cell_guid="98eb6dd9-8cb6-4e2d-9c00-9bc0486a0ef4" _uuid="2f97ac1d-8709-4663-a3cd-266f309495a0" jupyter={"outputs_hidden": false}
# ===================== smart_retriever.py =====================
"""Combines function retrieval (BM25 ± embeddings) + value retrieval (deterministic)."""

from __future__ import annotations

import re
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from rank_bm25 import BM25Okapi

# --- Date extraction (regex) ---

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
    # "tuần 22/2026", "thứ 5/2026" (week/ordinal, NOT month)
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
    # "2022-01-01 to 2022-12-31" (explicit ISO range)
    (
        r"(\d{4}-\d{2}-\d{2})\s*(?:den|đến|to|\-)\s*(\d{4}-\d{2}-\d{2})",
        lambda m: (m.group(1), m.group(2)),
    ),
]


def _last_day(year: int, month: int) -> str:
    """Last day of month."""
    import calendar
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last:02d}"


def _quarter_range(q: int, year: int) -> tuple[str, str]:
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return (f"{year}-{starts[q]}", f"{year}-{ends[q]}")


def extract_dates(query: str) -> dict[str, str]:
    """Extract from_date/to_date from Vietnamese query."""
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


# --- Data-level extraction (regex) ---

_DATA_LEVEL_PATTERNS = [
    (r"(?:theo|tong\s*hop)\s*(?:ngay|ngày|daily)", "day"),
    (r"(?:theo|tong\s*hop)\s*(?:tuan|tuần|weekly)", "week"),
    (r"(?:theo|tong\s*hop)\s*(?:thang|tháng|monthly)", "month"),
    (r"(?:theo|tong\s*hop)\s*(?:nam|năm|yearly)", "year"),
    (r"(?:hang|hàng)\s*(?:ngay|ngày)", "day"),
    (r"(?:hang|hàng)\s*(?:tuan|tuần)", "week"),
    (r"(?:hang|hàng)\s*(?:thang|tháng)", "month"),
    (r"(?:thang|tháng)\s*\d{1,2}", "month"),  # "tháng 6" → monthly
]


def extract_data_level(query: str) -> str | None:
    """Extract aggregation level from Vietnamese query."""
    query_norm = normalize_vietnamese(query)
    for pattern, level in _DATA_LEVEL_PATTERNS:
        if re.search(pattern, query_norm):
            return level
    return None


# Function retriever: BM25 over normalized Vietnamese descriptions. Hybrid mode needs encoder_model.

class FunctionRetriever:
    """BM25-based function retrieval with Vietnamese normalization. Hybrid mode needs encoder_model."""

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

        # Normalize everything
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
        """Build search string from function schema."""
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


# Argument value retriever: deterministic lookup + scoring

class ArgumentValueRetriever:
    """Deterministic value matching (no embeddings).
    Small enums → all. Large → alias + token match. Dates/data_level → regex."""

    def __init__(
        self,
        catalogs: dict[str, ValueCatalog],
        top_k_values: int = 5,
        include_all_threshold: int = 12,
    ):
        self.catalogs = catalogs
        self.top_k = top_k_values
        self.include_all_threshold = include_all_threshold

        # Schema param name → catalog key (may differ)
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
        """Find matching values per parameter."""
        result: dict[str, list[ValueMatch]] = {}
        params = function_schema.get("parameters", {})

        # Pre-normalize query once
        query_norm = expand_synonyms(normalize_vietnamese(query))
        query_tokens = tokenize_meaningful(query_norm)

        for param_name in params:
            # --- Special handling: dates ---
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

            # --- Special handling: data_level ---
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

            # --- Standard catalog lookup ---
            catalog = self._get_catalog(param_name)
            if catalog is None:
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
        """Merge value matches across candidate functions."""
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
        """Resolve param → catalog: direct → suffix → prefix fallback."""
        catalog_key = self._param_to_catalog.get(param_name)
        if catalog_key and catalog_key in self.catalogs:
            return self.catalogs[catalog_key]

        if param_name in self.catalogs:
            return self.catalogs[param_name]

        # "network_provider" → "provider"
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


# Combined TelcoRetriever

@dataclass
class RetrievalResult:
    function_names: list[str]
    argument_values: dict[str, list[ValueMatch]]
    extracted_dates: dict[str, str]
    extracted_data_level: str | None


class TelcoRetriever:
    """Combined function + argument value retriever."""

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

```python _cell_guid="5141fb81-8bd2-4bf5-ae45-4feee6401362" _uuid="15117073-e519-4131-83a5-4c494b77d0fd" jupyter={"outputs_hidden": false}
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

```python _cell_guid="ab94bc97-16bd-4e1a-8f61-5beb0d6b85bb" _uuid="f9fbcf95-19cc-4afa-8c88-61b0f501f5b2" jupyter={"outputs_hidden": false}
# ===================== base_reward.py =====================
import re
import json
from typing import Any

# Coerce ground_truth (JSON string or dict) into normalized dict.
# Avoids ArrowInvalid from pyarrow struct-schema inference.
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
    """Extract first <tool_call>...</tool_call> JSON."""
    match = _TOOL_CALL_RE.search(response)
    if not match:
        # fallback: old <call> format
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
    """Extract all <tool_call> blocks (for sequential workflows)."""
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


# Component reward functions

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


# Trajectory-level reward for RC-GRPO
# Uses the SAME {function, arguments} schema as dataset & extract_call

# Composite reward for GRPO training (works for both single & sequential)

def make_reward_function(
    ground_truth: dict,
    function_library: dict,
    sandbox_cls=None,
    weights: dict | None = None,
):
    """Build reward(response_text) -> float. Returns 1.0 for correct abstention."""
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
    """Build binary reward(response_text) -> int (0/1), thresholding coverage reward at >=1.0."""
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
        return 1 if compute_action_coverage_reward(agent_calls, gold_calls) >= 1.0 else 0

    return _reward


# TRL-compatible batch reward functions

def function_reward(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """F1-based function selection reward (penalises misses AND spurious calls)."""
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

        true_positives = sum(1 for af in agent_funcs if af in gold_funcs)
        precision = true_positives / len(agent_funcs)

        recall = sum(1 for gf in gold_funcs if gf in agent_funcs) / len(gold_funcs)

        if precision + recall == 0.0:
            rewards.append(0.0)
        else:
            rewards.append(2.0 * precision * recall / (precision + recall))
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Multi-component format reward: tag presence (0.3) + JSON parse (0.3) + clean output (0.4)."""
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
    """Per-parameter matching accuracy via compute_action_coverage_reward."""
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
    """Weighted sum: format + function + arguments (no reasoning term, per Eq. 5)."""
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

```python _cell_guid="45107bee-478b-4776-912f-b5ed3ca63701" _uuid="b2b72919-61f5-476c-a80c-9d17ab233bfb" jupyter={"outputs_hidden": false}
"""RC-GRPO (arXiv:2602.03025): reward-conditioned GRPO with Schulman k3 KL."""

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


HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"
REWARD_TOKENS = [HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN]


def compute_action_coverage_reward(
    agent_tool_calls: List[Dict],
    gold_tool_calls: List[Dict],
) -> float:
    """Per-parameter accuracy: fraction of gold params matched per call, averaged."""

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

```python _cell_guid="0dab777e-54c5-45e6-8790-3ea50d35de2e" _uuid="06332f7c-449c-45b9-8776-abb46dc8cce1" jupyter={"outputs_hidden": false}
# --- RC-GRPO ↔ TRL integration ---
# RCGRPOTrainer was referenced but never defined. Subclasses TRL's
# GRPOTrainer to inject reward tokens per rollout (Eq. 3-4); token is
# stripped before scoring — R(tau) only checks correctness (Eq. 5).

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
    """Inject `[Reward Goal: <token>]` into system message (Appendix B)."""
    out = []
    injected = False
    for msg in prompt_messages:
        if msg.get("role") == "system" and not injected:
            out.append({**msg, "content": f"{msg['content']}\n[Reward Goal: {reward_token}]"})
            injected = True
        else:
            out.append(msg)
    return out


# MMR diversity reweighting functions (from WeiKangda/MMR-GRPO, ACL 2026)


def mmr_reweight_original(rewards, embeddings, lambda_div=0.7):
    """MMR diversity reweighting (Balasubramanian, 2015; λ fixed)."""
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
    """Adaptive λ via rel_std with temperature."""
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
    """Zero-hyperparam MMR via sigmoid(σ)."""
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


class RCGRPOTrainer(GRPOTrainer):
    """Reward-conditioned rollout sampling (Eq. 3-4)."""

    def __init__(self, *args, high_reward_probability: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_reward_probability = high_reward_probability
        # diagnostics, mirrors paper's "advantage spread" tracking (Table 5)
        self._last_reward_tokens: List[str] = []

    @staticmethod
    def register_reward_tokens(tokenizer) -> int:
        """Add <|high_reward|> / <|low_reward|> as special tokens. Returns count added (0 if already present)."""
        existing = set(tokenizer.all_special_tokens)
        to_add = [t for t in REWARD_TOKENS if t not in existing]
        if not to_add:
            return 0
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        return num_added

    def _generate_and_score_completions(self, inputs):
        """Inject reward token into each of G prompt repeats before vLLM generation + scoring."""
        G = self.args.num_generations

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

        return super()._generate_and_score_completions(conditioned_inputs)


class MMRGRPOTrainer(RCGRPOTrainer):
    """MMR-GRPO diversity-aware reweighting (ACL 2026 Findings)."""

    def __init__(self, *args, mmr_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mmr_config = mmr_config or {}
        self._mmr_log = []  # per-step MMR diagnostics

    @staticmethod
    def _render_prompts(inputs, processing_class):
        """Replicate TRL's maybe_apply_chat_template (text + conversational)."""
        prompts_text = []
        for example in inputs:
            prompt = example["prompt"]
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
                prompts_text.append(processing_class.apply_chat_template(prompt, tokenize=False))
            else:
                prompts_text.append(prompt)
        return prompts_text

    def _generate_and_score_completions(self, inputs):
        """Full override: MMR inserted between reward and advantage computation (can't call super())."""
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = self._render_prompts(inputs, self.processing_class)
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        if self.args.use_vllm:
            # vLLM colocate path (trl v0.22.2: self.llm.generate)
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)
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

        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids.tolist()
        )
        rewards = rewards_per_func.sum(dim=1)

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

        advantages = rewards.view(-1, self.num_generations)
        advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / \
                     (advantages.std(dim=1, keepdim=True) + 1e-4)
        advantages = advantages.view(-1)

        with torch.no_grad():
            old_per_token_logps = self._get_per_token_logps(
                self.model, prompt_completion_ids, prompt_mask,
                prompt_completion_ids.size(1) - prompt_ids.size(1),
            )

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
    """GTPO: Conflict-aware gradient correction (Simoni et al., arXiv 2508.03772)."""

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


def rc_grpo_reward_func(completions: list[str], ground_truth: list, **kwargs) -> list[float]:
    """R(tau) = R_state * R_action (Eq. 5), binary. Ignores reward-conditioning token (Sec. 3.3)."""
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
    """Format shaping reward (kept separate so binary R(tau) for token quantization stays clean)."""
    return format_reward(completions, **kwargs)
```

```python _cell_guid="c4fa4f72-3197-467e-a3b2-46f5e258efa1" _uuid="c0e8dbfc-e8c0-4087-a46c-e8b13e91879b" jupyter={"outputs_hidden": false}
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

# Chat template — supports system, user, assistant, tool, retriever

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

    # Register reward tokens
    existing_special = tokenizer.additional_special_tokens or []
    tokens_to_add = [t for t in REWARD_TOKENS if t not in existing_special]
    if tokens_to_add:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_special + tokens_to_add}
        )
        print(f"[base_trainer] Added reward tokens: {tokens_to_add}")

    print("[base_trainer] Custom chat template registered (supports 'retriever' role).")


# SYSTEM PROMPT — updated to support both single and sequential workflows

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


# Prompt building helpers

def build_function_description(func_name: str, schema: dict) -> str:
    """Format function schema into markdown block."""
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
    """Build retriever context block with function descriptions + argument values."""
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
    load_in_4bit: bool = False,
    fast_inference: bool = False,
    adapter_model_path: str | None = None,
    mode: str = "train",
    lora_rank: int = 16,
    lora_target_modules: list[str] | None = None,
    lora_dropout: float = 0.0,
    env_name: str = "local",
    gpu_memory_utilization: float | None = None,
) -> tuple:
    """Unified model loader: base model ± LoRA adapter (4 branches)."""
    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    # Kaggle path override
    if env_name == "kaggle" and base_model_name == "unsloth/Qwen3-4B-Instruct-2507":
        base_model_name = "/kaggle/input/models/dzung271828/unsloth/transformers/default/6/qwen3-4b-instruct-2507/qwen3-4b-instruct-2507"

    # Guard: don't pass adapter as base model
    if os.path.isdir(base_model_name) and os.path.exists(
        os.path.join(base_model_name, "adapter_config.json")
    ):
        raise ValueError(
            f"The path '{base_model_name}' appears to be a LoRA adapter "
            "(contains adapter_config.json), not a base model. "
            "Set `adapter_model_path=` to this path and `base_model_name=` "
            "to the original base model (e.g. 'unsloth/Qwen3-4B-Instruct-2507')."
        )

    if adapter_model_path is not None:
        print(f"[load_model] base_model_name={base_model_name}, adapter_model_path={adapter_model_path}")
    else:
        print(f"[load_model] base_model_name={base_model_name}")

    if adapter_model_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model (fast_inference handles GRPO patching)
    kwargs = dict(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        dtype=None,
        gpu_memory_utilization=gpu_memory_utilization if gpu_memory_utilization is not None else (0.3 if os.environ.get("UNSLOTH_VLLM_STANDBY", "0") != "1" else 0.8),
    )
    if mode == "train":
        kwargs["max_lora_rank"] = lora_rank

    model, _ = FastLanguageModel.from_pretrained(**kwargs)

    if adapter_model_path is not None:
        # Checkpoint resume: create PeftModel then load adapter
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

    if mode == "inference":
        FastLanguageModel.for_inference(model)

    return model, tokenizer


# GRPO config builder — asymmetric clip only (CISPO/DAPO/GT), not RC-GRPO

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
        # --- vLLM KV cache reduction ---
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


# Dataset loaders

def execution_reward(
    completions: list[str],
    ground_truth: list,
    function_library: dict,
    **kwargs,
) -> list[float]:
    """Execution-based R_state reward, continuous in [0, 1]. Runs extracted calls through Sandbox."""
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
    """TRL-compatible reward function list for selected algorithm."""
    if algorithm == "rc_grpo":
        # Same reward functions as vanilla GRPO — diff is prompt conditioning only
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

# Reward-token injection helper for TRL GRPOTrainer subclass

def inject_reward_token_into_prompt(
    prompt: str,
    reward_token: str,
) -> str:
    """Inject [Reward Goal: <token>] after system content (before <|im_end|>)."""
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

```python _cell_guid="655941c3-02d4-4086-bad6-4fc67123114f" _uuid="3a478735-5665-4570-ad10-0a86bd029e21" jupyter={"outputs_hidden": false}
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
    model_path: str,
    test_dataset_path: str,
    function_library: dict,
    retriever: FunctionRetriever,
    sandbox: Sandbox,
    top_k: int = 5,
    max_new_tokens: int = 2048,
    model_name_tag: str = "model",
    use_dataset_retrieval: bool = True,
    argument_values: dict | None = None,
    condition_on_high_reward: bool = True,
    use_vllm: bool = True,
    batch_size: int = 32,
    gpu_memory_utilization: float | None = None,
    base_model_name: str | None = None,
) -> dict:
    """Unified evaluation: batched vLLM or serial HF generate. gpu_memory_utilization defaults to config."""
    logger = get_logger(__name__)
    tag = "vLLM-Bench" if use_vllm else "Benchmark"
    logger.info(f"[{tag}] Loading model from {model_path}")

    if gpu_memory_utilization is None:
        gpu_memory_utilization = TRAIN_CONFIG["model"]["gpu_memory_utilization"]

    if base_model_name is None:
        base_model_name = TRAIN_CONFIG["model"]["name"]

    model, tokenizer = load_model(
        base_model_name=base_model_name,
        adapter_model_path=model_path,
        mode="inference",
        load_in_4bit=False,
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

            # Condition on <|high_reward|> at inference (only if reward tokens registered)
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

    # Self-cleanup: release GPU memory after each checkpoint
    # (vLLM workers persist across calls → OOM without explicit shutdown)
    try:
        if use_vllm and hasattr(model, 'llm') and model.llm is not None:
            llm = model.llm
            if hasattr(llm, 'shutdown') and callable(llm.shutdown):
                llm.shutdown()
            elif hasattr(llm, 'sleep') and callable(llm.sleep):
                llm.sleep(level=2)
            logger.info(f"[{tag}] Internal vLLM engine shut down")
    except Exception as exc:
        logger.warning(f"[{tag}] vLLM engine shutdown failed: {exc}")

    try:
        from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass

    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass

    import gc as _gc
    for _ in range(3):
        _gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    logger.info(f"[{tag}] GPU memory released")

    return {"model": model_name_tag, "per_sample": results, "aggregate": agg}
```

```python _cell_guid="65274daf-e3ec-4f60-99c2-ea73f4ff67a1" _uuid="97b9e4d7-78d6-41f7-915a-0ddcfe0c62d8" jupyter={"outputs_hidden": false}
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

```python _cell_guid="cleanup-training-001" _uuid="cleanup-training-001" jupyter={"outputs_hidden": false}
# ===================== GPU cleanup utility =====================
def cleanup_training(trainer=None, model=None, tokenizer=None, logger=None):
    """Release ALL training resources so evaluation starts clean. Idempotent."""
    _log = logger.info if hasattr(logger, 'info') else print

    # Shutdown vLLM engine
    if trainer is not None and hasattr(trainer, 'llm') and trainer.llm is not None:
        try:
            llm = trainer.llm
            if hasattr(llm, 'shutdown') and callable(llm.shutdown):
                llm.shutdown()
                _log("[cleanup] vLLM LLM.shutdown() completed")
            else:
                if hasattr(llm, 'sleep') and callable(llm.sleep):
                    llm.sleep(level=2)
                    _log("[cleanup] vLLM sleep(level=2) completed")
        except Exception as exc:
            _log(f"[cleanup] WARNING: vLLM shutdown failed: {exc}")

    # Destroy vLLM distributed state
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

    # Destroy torch distributed process group
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as exc:
        _log(f"[cleanup] WARNING: destroy_process_group failed: {exc}")

    # Unfreeze GC heap (vLLM V1 freezes startup objects)
    try:
        import gc as _gc
        _gc.unfreeze()
    except Exception:
        pass

    # Delete trainer
    if trainer is not None:
        try:
            if hasattr(trainer, 'model') and trainer.model is not None:
                if hasattr(trainer.model, '_delete_llm'):
                    trainer.model._delete_llm()
        except Exception:
            pass
        try:
            del trainer
        except Exception:
            pass

    # Delete model and tokenizer
    if model is not None:
        try:
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

    # Garbage collection
    import gc as _gc2
    for _ in range(3):
        _gc2.collect()

    # CUDA cache reset
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _log(f"[cleanup] GPU memory freed. "
         f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
         f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
```

```python _cell_guid="3eeb6334-7eb7-4716-8d6f-10515dd86d9b" _uuid="889bd550-e2f2-47dc-9254-d4565def65cf" jupyter={"outputs_hidden": false}
import os
if ENV_NAME == "colab":
    # Create the data directory
    os.makedirs('/content/data', exist_ok=True)
    # Unzip the file into the data folder
    !unzip -o /content/generated.zip -d /content/ToolFormer/data
```

```python _cell_guid="9fd3dde9-c452-4509-8ff2-3f7bb9196318" _uuid="b1dfcfea-fb42-49e2-8b31-6e2871878c2b" jupyter={"outputs_hidden": false}
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
        "name": "unsloth/Qwen3-4B-Instruct-2507",
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
        "eval_batch_size": 64,
    },
    "grpo": {
        "temperature": 1.0,
        "epsilon": 0.2,            # Table 10: symmetric Clip Ratio = 0.2
        "kl_coefficient": float(os.environ.get("KL_COEFFICIENT", "0.0")),
                                   # Override via env var (default 0.0 for
                                   # reference-free comparison across all
                                   # 4 loss types).  Also changeable at
                                   # runtime via TRAIN_CONFIG.
        "loss_type": "grpo",     # Options: "grpo", "dapo", "cispo", "gtpo", "mmr_grpo"
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
# Load function library + argument values
function_library = load_function_library(FUNCTION_LIBRARY_PATH)
print(f"Loaded {len(function_library)} functions")

argument_values_catalog = None
if ARGUMENT_VALUES_PATH.exists():
    argument_values_catalog = load_catalog_from_json(str(ARGUMENT_VALUES_PATH))
    print(f"Loaded argument values catalog with {len(argument_values_catalog)} parameters")
else:
    print("No argument_values.json; argument value retrieval will be skipped.")

# ===================== Load model & register reward tokens =====================
MODE_OUTPUT_DIR = TRAIN_CONFIG["training"]["output_dir"]
```

```python _cell_guid="4c336369-3996-4d38-b3eb-fc02ef64ccac" _uuid="17676702-da3f-4dc9-a424-ae393540814e" jupyter={"outputs_hidden": false}
# Smart truncation safety net (legacy, removed — all samples fit 8192)
```

```python _cell_guid="e6f25d35-a8fd-4a73-b2d9-8b3c9ee8b1c7" _uuid="874d6485-b09b-4752-b118-99d722f6c98c" jupyter={"outputs_hidden": false}
# # ===================== SFT TRAINING =====================
# def load_jsonl(path: str) -> list[dict]:
#     samples: list[dict] = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 samples.append(json.loads(line))
#     return samples

# if MODE == "sft":
#     print("\n" + "=" * 70)
#     print("SFT: Supervised Fine-Tuning on expert demonstrations")
#     print("=" * 70)
#     model, tokenizer = load_model(
#         base_model_name=TRAIN_CONFIG["model"]["name"],
#         max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],
#         load_in_4bit=TRAIN_CONFIG["model"]["load_in_4bit"],
#         fast_inference=False,
#         adapter_model_path=None,
#         mode="train",
#         lora_rank=TRAIN_CONFIG["lora"]["r"],
#         lora_target_modules=TRAIN_CONFIG["lora"]["target_modules"],
#         lora_dropout=TRAIN_CONFIG["lora"]["lora_dropout"],
#         env_name=ENV_NAME,
#     )
#     print("Model and tokenizer loaded.")

#     print("Tokenizer special tokens:")
#     print(f"  bos_token: {tokenizer.bos_token}")
#     print(f"  eos_token: {tokenizer.eos_token}")
#     print(f"  pad_token: {tokenizer.pad_token}")
#     print(f"  additional_special_tokens: {tokenizer.additional_special_tokens}")

#     sft_cfg = TRAIN_CONFIG["sft"]
#     raw_records = load_jsonl(TRAIN_CONFIG["data"]["sft_path"])
#     from datasets import Dataset

#     # Use full text field directly; train_on_responses_only will mask
#     # everything except the assistant response.
#     dataset = Dataset.from_list(raw_records)

#     from trl import SFTTrainer, SFTConfig

#     training_args = SFTConfig(
#         output_dir=sft_cfg["output_dir"],
#         per_device_train_batch_size=sft_cfg["batch_size"],
#         gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
#         num_train_epochs=sft_cfg["num_epochs"],
#         learning_rate=sft_cfg["learning_rate"],
#         max_seq_length=sft_cfg["max_seq_length"],
#         max_grad_norm=1.0,
#         weight_decay=0.001,
#         optim="adamw_8bit",
#         seed=3407,
#         warmup_ratio=0.1,
#         logging_steps=5,
#         save_strategy="epoch",
#         save_total_limit=1,
#         report_to="none",
#         fp16=not torch.cuda.is_bf16_supported(),
#         bf16=torch.cuda.is_bf16_supported(),
#         dataloader_num_workers=4,
#         dataset_text_field="text",
#     )

#     trainer = SFTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         train_dataset=dataset,
#     )

#     # Use Unsloth's train_on_responses_only to mask prompt tokens,
#     # so the model only learns from the assistant response.
#     from unsloth.chat_templates import train_on_responses_only
#     trainer = train_on_responses_only(
#         trainer,
#         instruction_part="<|im_start|>user\n",
#         response_part="<|im_start|>assistant\n",
#     )

#     print("=== SFT dataset sample ===")
#     for k, v in dataset[0].items():
#         print(f"  {k}: {str(v)}")
#     print("===========================")

#     # ── Hyperparameter snapshot ──
#     _sft_table = [
#         ["Mode", "SFT - Supervised Fine-Tuning"],
#         ["Dataset", TRAIN_CONFIG["data"]["sft_path"]],
#         ["Samples", len(dataset)],
#         ["Base model", TRAIN_CONFIG["model"]["name"]],
#         ["Max seq length", TRAIN_CONFIG["model"]["max_seq_length"]],
#         ["Load in 4bit", TRAIN_CONFIG["model"]["load_in_4bit"]],
#         ["Full finetuning", TRAIN_CONFIG["model"]["full_finetuning"]],
#         ["LoRA rank (r)", TRAIN_CONFIG["lora"]["r"]],
#         ["LoRA dropout", TRAIN_CONFIG["lora"]["lora_dropout"]],
#         ["LoRA targets", TRAIN_CONFIG["lora"]["target_modules"]],
#         ["LoRA bias", TRAIN_CONFIG["lora"]["bias"]],
#         ["Gradient checkpointing", TRAIN_CONFIG["lora"]["use_gradient_checkpointing"]],
#         ["Learning rate", training_args.learning_rate],
#         ["Batch size (per device)", training_args.per_device_train_batch_size],
#         ["Gradient accumulation steps", training_args.gradient_accumulation_steps],
#         ["Num epochs", training_args.num_train_epochs],
#         ["Max grad norm", training_args.max_grad_norm],
#         ["Weight decay", training_args.weight_decay],
#         ["Optimizer", training_args.optim],
#         ["Warmup ratio", training_args.warmup_ratio],
#         ["Seed", training_args.seed],
#         ["Logging steps", training_args.logging_steps],
#         ["Save strategy", training_args.save_strategy],
#         ["Save total limit", training_args.save_total_limit],
#         ["Dataloader workers", training_args.dataloader_num_workers],
#         ["FP16", training_args.fp16],
#         ["BF16", training_args.bf16],
#         ["Precision", "bf16" if torch.cuda.is_bf16_supported() else "fp16"],
#     ]
#     print(tabulate(_sft_table, headers=["Hyperparameter", "Value"], tablefmt="fancy_grid"))

#     trainer.train()
#     trainer.save_model(sft_cfg["output_dir"])
#     tokenizer.save_pretrained(Path(sft_cfg["output_dir"]))

#     # ── Save hyperparameters ──
#     _sft_hp = {
#         "mode": "sft",
#         "timestamp": datetime.datetime.now().isoformat(),
#         "dataset": {
#             "path": TRAIN_CONFIG["data"]["sft_path"],
#             "samples": len(dataset),
#         },
#         "model": {
#             "base_model": TRAIN_CONFIG["model"]["name"],
#             "max_seq_length": TRAIN_CONFIG["model"]["max_seq_length"],
#             "load_in_4bit": TRAIN_CONFIG["model"]["load_in_4bit"],
#             "full_finetuning": TRAIN_CONFIG["model"]["full_finetuning"],
#         },
#         "lora": {
#             "r": TRAIN_CONFIG["lora"]["r"],
#             "lora_dropout": TRAIN_CONFIG["lora"]["lora_dropout"],
#             "target_modules": TRAIN_CONFIG["lora"]["target_modules"],
#             "bias": TRAIN_CONFIG["lora"]["bias"],
#             "use_gradient_checkpointing": TRAIN_CONFIG["lora"]["use_gradient_checkpointing"],
#         },
#         "training": {
#             "learning_rate": training_args.learning_rate,
#             "per_device_train_batch_size": training_args.per_device_train_batch_size,
#             "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
#             "num_train_epochs": training_args.num_train_epochs,
#             "max_grad_norm": training_args.max_grad_norm,
#             "weight_decay": training_args.weight_decay,
#             "optim": training_args.optim,
#             "warmup_ratio": training_args.warmup_ratio,
#             "seed": training_args.seed,
#             "logging_steps": training_args.logging_steps,
#             "save_strategy": training_args.save_strategy,
#             "save_total_limit": training_args.save_total_limit,
#             "dataloader_num_workers": training_args.dataloader_num_workers,
#         },
#         "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
#     }
#     _sft_out = Path(sft_cfg["output_dir"])
#     _sft_out.mkdir(parents=True, exist_ok=True)
#     with open(_sft_out / "training_hyperparameters.json", "w") as f:
#         json.dump(_sft_hp, f, indent=2, ensure_ascii=False)

#     print(f"[SFT] Model saved -> {sft_cfg['output_dir']}")
```

```python _cell_guid="a93fa11f-e8b4-45d3-8f90-606c4a38b686" _uuid="ca5d8248-6cd5-4808-a7ce-dcf80350f593" jupyter={"outputs_hidden": false}
# if MODE == "sft":
#     !zip -r sft_model.zip outputs/sft_model
#     from IPython.display import FileLink
#     FileLink("sft_model.zip")
```

```python _cell_guid="49353534-fc16-474f-9f8f-ca6a0bac2ebd" _uuid="f6600845-eb26-49d5-a8de-19608d61d10a" jupyter={"outputs_hidden": false}
# # ===================== STAGE 1: RCTP-FT (only for rc_grpo) =====================
# SAMPLE_LOG = get_logger("sample")
# high_reward_probability = 0.5  # overwritten below from real data stats

# if MODE == "rctp_ft":
#     print("\n" + "=" * 70)
#     print("STAGE 1: Reward-Conditioned Trajectory Policy (RCTP) Fine-tuning")
#     print("=" * 70)
#     model, tokenizer = load_model(
#         base_model_name=TRAIN_CONFIG["model"]["name"],
#         max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],
#         load_in_4bit=TRAIN_CONFIG["model"]["load_in_4bit"],
#         fast_inference=False,
#         adapter_model_path=None,
#         mode="train",
#         lora_rank=TRAIN_CONFIG["lora"]["r"],
#         lora_target_modules=TRAIN_CONFIG["lora"]["target_modules"],
#         lora_dropout=TRAIN_CONFIG["lora"]["lora_dropout"],
#         env_name=ENV_NAME,
#     )
#     print("Model and tokenizer loaded.")

#     print("Tokenizer special tokens:")
#     print(f"  bos_token: {tokenizer.bos_token}")
#     print(f"  eos_token: {tokenizer.eos_token}")
#     print(f"  pad_token: {tokenizer.pad_token}")
#     print(f"  additional_special_tokens: {tokenizer.additional_special_tokens}")

#     rctp_cfg = TRAIN_CONFIG["rctp_ft"]

#     # Inline: Trajectory dataclass + helper (standalone, no scripts/ dependency)
#     def binary_reward_to_token(reward: int) -> str:
#         return HIGH_REWARD_TOKEN if reward == 1 else LOW_REWARD_TOKEN

#     @dataclass
#     class Trajectory:
#         prompt_messages: list[dict[str, str]]
#         response_text: str
#         reward: int
#         reward_token: str = field(init=False)

#         def __post_init__(self):
#             self.reward_token = binary_reward_to_token(self.reward)

#     rctp_records = load_jsonl(TRAIN_CONFIG["data"]["rctp_path"])
#     rctp_trajectories = [Trajectory(**rec) for rec in rctp_records]

#     # Eq. 3: p should match the RCTP-FT dataset's empirical success rate
#     n_success = sum(1 for t in rctp_trajectories if t.reward == 1)
#     n_total = len(rctp_trajectories)
#     high_reward_probability = n_success / max(1, n_total)
#     print(f"[Stage 1] p (high_reward_probability for Eq. 3) = {high_reward_probability:.4f} "
#           f"({n_success}/{n_total} success)")

#     # ── Stage 1 via SFTTrainer ──
#     from datasets import Dataset
#     from trl import SFTTrainer, SFTConfig

#     def _format_trajectory(trajectory):
#         """
#         Build reward-conditioned text: inject [Reward Goal: <token>] into the
#         system message (Appendix B), then append the assistant response.
#         """
#         reward_prefix = f"[Reward Goal: {trajectory.reward_token}]"
#         messages = []
#         injected = False
#         for msg in trajectory.prompt_messages:
#             if msg["role"] == "system" and not injected:
#                 messages.append(
#                     {"role": "system", "content": f"{msg['content']}\n{reward_prefix}"}
#                 )
#                 injected = True
#             else:
#                 messages.append(msg)
#         messages.append({"role": "assistant", "content": trajectory.response_text})
#         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

#     formatted_texts = [_format_trajectory(t) for t in rctp_trajectories]
#     train_dataset = Dataset.from_list([{"text": t} for t in formatted_texts])

#     training_args = SFTConfig(
#         output_dir=rctp_cfg["output_dir"],
#         per_device_train_batch_size=rctp_cfg["batch_size"],
#         gradient_accumulation_steps=rctp_cfg["gradient_accumulation_steps"],
#         num_train_epochs=rctp_cfg["num_epochs"],
#         learning_rate=rctp_cfg["learning_rate"],
#         max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],  # ← was 1792 (too short for prompt+completion)
#         max_grad_norm=1.0,
#         weight_decay=0.001,
#         optim="adamw_8bit",
#         seed=3407,
#         warmup_ratio=0.1,
#         logging_steps=5,
#         save_strategy="epoch",
#         save_total_limit=1,
#         report_to="none",
#         remove_unused_columns=True,
#         fp16=not torch.cuda.is_bf16_supported(),
#         bf16=torch.cuda.is_bf16_supported(),
#         dataloader_num_workers=4,
#         dataset_text_field="text",
#     )

#     trainer = SFTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         train_dataset=train_dataset,
#     )

#     # Use Unsloth's train_on_responses_only to mask prompt tokens,
#     # so the model only learns from the assistant response.
#     from unsloth.chat_templates import train_on_responses_only
#     trainer = train_on_responses_only(
#         trainer,
#         instruction_part="<|im_start|>user\n",
#         response_part="<|im_start|>assistant\n",
#     )

#     print("=== RCTP dataset sample ===")
#     for k, v in train_dataset[0].items():
#         print(f"  {k}: {str(v)}")
#     print("===========================")

#     SAMPLE_LOG.info("stage=pre_stage1_train type=summary trajectories=%d epochs=%d batch_size=%d",
#                     len(rctp_trajectories), training_args.num_train_epochs, training_args.per_device_train_batch_size)

#     # ── Hyperparameter snapshot ──
#     _rctp_table = [
#         ["Mode", "RCTP-FT (Stage 1)"],
#         ["Dataset", TRAIN_CONFIG["data"]["rctp_path"]],
#         ["Trajectories", len(rctp_trajectories)],
#         ["High reward probability", high_reward_probability],
#         ["Base model", TRAIN_CONFIG["model"]["name"]],
#         ["Max seq length", TRAIN_CONFIG["model"]["max_seq_length"]],
#         ["Load in 4bit", TRAIN_CONFIG["model"]["load_in_4bit"]],
#         ["LoRA rank (r)", TRAIN_CONFIG["lora"]["r"]],
#         ["LoRA dropout", TRAIN_CONFIG["lora"]["lora_dropout"]],
#         ["LoRA targets", TRAIN_CONFIG["lora"]["target_modules"]],
#         ["LoRA bias", TRAIN_CONFIG["lora"]["bias"]],
#         ["Gradient checkpointing", TRAIN_CONFIG["lora"]["use_gradient_checkpointing"]],
#         ["Learning rate", training_args.learning_rate],
#         ["Batch size (per device)", training_args.per_device_train_batch_size],
#         ["Gradient accumulation steps", training_args.gradient_accumulation_steps],
#         ["Num epochs", training_args.num_train_epochs],
#         ["Max grad norm", training_args.max_grad_norm],
#         ["Weight decay", training_args.weight_decay],
#         ["Optimizer", training_args.optim],
#         ["Warmup ratio", training_args.warmup_ratio],
#         ["Seed", training_args.seed],
#         ["Logging steps", training_args.logging_steps],
#         ["Save strategy", training_args.save_strategy],
#         ["Save total limit", training_args.save_total_limit],
#         ["Dataloader workers", training_args.dataloader_num_workers],
#         ["FP16", training_args.fp16],
#         ["BF16", training_args.bf16],
#         ["Precision", "bf16" if torch.cuda.is_bf16_supported() else "fp16"],
#     ]
#     print(tabulate(_rctp_table, headers=["Hyperparameter", "Value"], tablefmt="fancy_grid"))

#     trainer.train()
#     trainer.save_model(rctp_cfg["output_dir"])
#     tokenizer.save_pretrained(Path(rctp_cfg["output_dir"]))

#     # ── Save hyperparameters ──
#     _rctp_hp = {
#         "mode": "rctp_ft",
#         "timestamp": datetime.datetime.now().isoformat(),
#         "dataset": {
#             "path": TRAIN_CONFIG["data"]["rctp_path"],
#             "trajectories": len(rctp_trajectories),
#             "high_reward_probability": high_reward_probability,
#         },
#         "model": {
#             "base_model": TRAIN_CONFIG["model"]["name"],
#             "max_seq_length": TRAIN_CONFIG["model"]["max_seq_length"],
#             "load_in_4bit": TRAIN_CONFIG["model"]["load_in_4bit"],
#         },
#         "lora": {
#             "r": TRAIN_CONFIG["lora"]["r"],
#             "lora_dropout": TRAIN_CONFIG["lora"]["lora_dropout"],
#             "target_modules": TRAIN_CONFIG["lora"]["target_modules"],
#             "bias": TRAIN_CONFIG["lora"]["bias"],
#             "use_gradient_checkpointing": TRAIN_CONFIG["lora"]["use_gradient_checkpointing"],
#         },
#         "training": {
#             "learning_rate": training_args.learning_rate,
#             "per_device_train_batch_size": training_args.per_device_train_batch_size,
#             "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
#             "num_train_epochs": training_args.num_train_epochs,
#             "max_grad_norm": training_args.max_grad_norm,
#             "weight_decay": training_args.weight_decay,
#             "optim": training_args.optim,
#             "warmup_ratio": training_args.warmup_ratio,
#             "seed": training_args.seed,
#             "logging_steps": training_args.logging_steps,
#             "save_strategy": training_args.save_strategy,
#             "save_total_limit": training_args.save_total_limit,
#             "dataloader_num_workers": training_args.dataloader_num_workers,
#         },
#         "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
#     }
#     _rctp_out = Path(rctp_cfg["output_dir"])
#     _rctp_out.mkdir(parents=True, exist_ok=True)
#     with open(_rctp_out / "training_hyperparameters.json", "w") as f:
#         json.dump(_rctp_hp, f, indent=2, ensure_ascii=False)

#     print(f"[Stage 1] RCTP-FT checkpoint saved -> {rctp_cfg['output_dir']}")
#     print("(This checkpoint becomes pi_ref for Stage 2, per Algorithm 1 line 9.)")
```

```python _cell_guid="a860c263-9dc0-4306-8610-540b6ec5f402" _uuid="890742eb-a63b-4307-b16b-41e46163ce90" jupyter={"outputs_hidden": false}
# if MODE == "rctp_ft":
#     !zip -r rctp_ft_model.zip outputs/rctp_ft_model
#     from IPython.display import FileLink
#     FileLink("rctp_ft_model.zip")
```

```python _cell_guid="a4804ae9-0772-4245-ad87-1d020fdaca27" _uuid="517691a9-c0e7-4fb6-9a53-29904d922999" jupyter={"outputs_hidden": false}
# ===================== Unified model loader for Stage 2 =====================
# Loads a base model with a fine-tuned LoRA adapter for inference or training,
# controlled by the `mode` argument.  Named parameters base_model_path /
# adapter_model_path make it clear what each path refers to.
# (Now handled by the unified load_model() above.)
```

```python _cell_guid="46ac1b06-e4aa-4d50-8ead-140138c8d50c" _uuid="52a8db35-6607-4704-ba46-608e59108903" jupyter={"outputs_hidden": false}
# if MODE in ("grpo", "rc_grpo"):
#     # ===================== STAGE 2: RC-GRPO (or vanilla GRPO baseline) =====================
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"  # required by vLLM standby mode 
#     os.environ["UNSLOTH_VLLM_STANDBY"] = "0"  # vLLM standby mode — shares GPU with trainer
#     print("\n" + "=" * 70)
#     print(f"STAGE 2: {'RC-GRPO' if MODE == 'rc_grpo' else 'Vanilla GRPO baseline'}")
#     print("=" * 70)

#     # ── Set model paths for Stage 2 ──────────────────────────────────
#     # The adapter checkpoint is typically the Stage 1 (RCTP-FT) output.
#     # Adjust these to point to your trained checkpoint directories.
#     base_model_path = "unsloth/Qwen3-4B-Instruct-2507"
#     adapter_model_path = "outputs/rctp_ft_model"  # Stage 1 (RCTP-FT) output

#     print(f"[Stage 2] Loading base_model_path = {base_model_path}")
#     print(f"[Stage 2] Loading adapter_model_path = {adapter_model_path}")

#     model, tokenizer = load_model(
#         base_model_name=base_model_path,
#         adapter_model_path=adapter_model_path,
#         mode="train",          # keep in train mode for further fine-tuning
#         fast_inference=True,
#         max_seq_length=TRAIN_CONFIG["model"]["max_seq_length"],
#         load_in_4bit=True,
#         env_name=ENV_NAME,
#     )
#     print("Model and tokenizer loaded.")
#     print("Tokenizer special tokens:")
#     print(f"  bos_token: {tokenizer.bos_token}")
#     print(f"  eos_token: {tokenizer.eos_token}")
#     print(f"  pad_token: {tokenizer.pad_token}")
#     print(f"  additional_special_tokens: {tokenizer.additional_special_tokens}")

#     from datasets import Dataset
#     if MODE == "rc_grpo":
#         data_path = TRAIN_CONFIG["data"].get("rcgrpo_path", TRAIN_CONFIG["data"]["grpo_path"])
#     else:
#         data_path = TRAIN_CONFIG["data"]["grpo_path"]
#     dataset = Dataset.from_list(load_jsonl(data_path))

#     grpo_args = build_grpo_config(TRAIN_CONFIG, output_dir=TRAIN_CONFIG["training"]["output_dir"])

#     if MODE == "rc_grpo":
#         loss_type = TRAIN_CONFIG["grpo"]["loss_type"]
#         shared_kwargs = dict(
#             model=model,
#             processing_class=tokenizer,
#             reward_funcs=build_trl_reward_functions("rc_grpo"),
#             args=grpo_args,
#             train_dataset=dataset,
#             high_reward_probability=high_reward_probability,
#         )
#         if loss_type == "gtpo":
#             trainer = GTPOTrainer(
#                 **shared_kwargs,
#                 entropy_threshold=TRAIN_CONFIG["gtpo"]["entropy_threshold"],
#                 gamma=TRAIN_CONFIG["gtpo"]["gamma"],
#             )
#         elif loss_type == "mmr_grpo":
#             trainer = MMRGRPOTrainer(
#                 **shared_kwargs,
#                 mmr_config=TRAIN_CONFIG["mmr"],
#             )
#         else:
#             trainer = RCGRPOTrainer(**shared_kwargs)
#     elif MODE == "grpo":
#         trainer = GRPOTrainer(
#             model=model,
#             processing_class=tokenizer,
#             reward_funcs=build_trl_reward_functions("grpo"),
#             args=grpo_args,
#             train_dataset=dataset,
#         )

#     print(f"Trainer initialized for {MODE}")
#     print("=== GRPO dataset sample ===")
#     for k, v in dataset[0].items():
#         print(f"  {k}: {str(v)}")
#     print("===========================")

#     custom_tags = ["<reasoning>", "</reasoning>", "<tool_call>", "</tool_call>", HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN]
#     existing = set(tokenizer.all_special_tokens)
#     conflicts = [tag for tag in custom_tags if tag in existing and tag not in REWARD_TOKENS]
#     if conflicts:
#         print(f"WARNING: custom tags collide with existing special tokens: {conflicts}")
#     else:
#         print("No conflicts detected with custom tags.")
#     SAMPLE_LOG.info("stage=pre_stage2_train type=summary algorithm=%s dataset_size=%d generations=%d",
#                     MODE, len(dataset), grpo_args.num_generations)

#     # ── Hyperparameter snapshot ──
#     _grpo_table = [
#         ["Algorithm", f"RC-{TRAIN_CONFIG['grpo']['loss_type'].upper()}" if MODE == "rc_grpo" else "Vanilla GRPO"],
#         ["Trainer", trainer.__class__.__name__],
#         ["Dataset", data_path],
#         ["Samples", len(dataset)],
#         ["Base model", base_model_path],
#         ["Adapter model", adapter_model_path],
#         ["Base model name", TRAIN_CONFIG["model"]["name"]],
#         ["Max seq length", TRAIN_CONFIG["model"]["max_seq_length"]],
#         ["Load in 4bit", TRAIN_CONFIG["model"]["load_in_4bit"]],
#         ["GPU memory util (Unsloth)", TRAIN_CONFIG["model"]["gpu_memory_utilization"]],
#         ["Full finetuning", TRAIN_CONFIG["model"]["full_finetuning"]],
#         ["LoRA rank (r)", TRAIN_CONFIG["lora"]["r"]],
#         ["LoRA dropout", TRAIN_CONFIG["lora"]["lora_dropout"]],
#         ["LoRA targets", TRAIN_CONFIG["lora"]["target_modules"]],
#         ["LoRA bias", TRAIN_CONFIG["lora"]["bias"]],
#         ["Learning rate", TRAIN_CONFIG["training"]["learning_rate"]],
#         ["Adam beta1", TRAIN_CONFIG["training"]["adam_beta1"]],
#         ["Adam beta2", TRAIN_CONFIG["training"]["adam_beta2"]],
#         ["Weight decay", TRAIN_CONFIG["training"]["weight_decay"]],
#         ["Warmup ratio", TRAIN_CONFIG["training"]["warmup_ratio"]],
#         ["LR scheduler", TRAIN_CONFIG["training"]["lr_scheduler_type"]],
#         ["Optimizer", TRAIN_CONFIG["training"]["optim"]],
#         ["Batch size (per device)", TRAIN_CONFIG["training"]["per_device_train_batch_size"]],
#         ["Gradient accumulation steps", TRAIN_CONFIG["training"]["gradient_accumulation_steps"]],
#         ["Num generations", TRAIN_CONFIG["training"]["num_generations"]],
#         ["Max steps", TRAIN_CONFIG["training"]["max_steps"]],
#         ["Save steps", TRAIN_CONFIG["training"]["save_steps"]],
#         ["Logging steps", TRAIN_CONFIG["training"]["logging_steps"]],
#         ["Max grad norm", TRAIN_CONFIG["training"]["max_grad_norm"]],
#         ["Seed", TRAIN_CONFIG["training"]["seed"]],
#         ["Temperature", TRAIN_CONFIG["grpo"]["temperature"]],
#         ["Epsilon (clip)", TRAIN_CONFIG["grpo"]["epsilon"]],
#         ["Epsilon high", TRAIN_CONFIG["grpo"].get("epsilon_high", "N/A")],
#         ["KL coefficient (beta)", TRAIN_CONFIG["grpo"]["kl_coefficient"]],
#         ["Loss type", TRAIN_CONFIG["grpo"]["loss_type"]],
#         ["MMR enabled", TRAIN_CONFIG["mmr"]["enabled"]],
#         ["MMR mode", TRAIN_CONFIG["mmr"]["mode"]],
#         ["MMR lambda", TRAIN_CONFIG["mmr"]["lambda_div"]],
#         ["Mask truncated", TRAIN_CONFIG["grpo"]["mask_truncated_completions"]],
#         ["Max prompt length", TRAIN_CONFIG["data"]["max_prompt_length"]],
#         ["Max completion length", TRAIN_CONFIG["data"]["max_completion_length"]],
#         ["Precision", "bf16" if torch.cuda.is_bf16_supported() else "fp16"],
#     ]
#     print(tabulate(_grpo_table, headers=["Hyperparameter", "Value"], tablefmt="fancy_grid"))

#     print("Starting training...")
#     trainer.train()
#     print("Training completed.")

#     # ── Auto-save model and tokenizer after training ──
#     output_dir = MODE_OUTPUT_DIR
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)

#     # ── Save hyperparameters ──
#     _grpo_hp = {
#         "mode": MODE,
#         "timestamp": datetime.datetime.now().isoformat(),
#         "dataset": {
#             "path": data_path,
#             "samples": len(dataset),
#         },
#         "model": {
#             "base_model": TRAIN_CONFIG["model"]["name"],
#             "base_model_path": base_model_path,
#             "adapter_model_path": adapter_model_path,
#             "max_seq_length": TRAIN_CONFIG["model"]["max_seq_length"],
#             "load_in_4bit": TRAIN_CONFIG["model"]["load_in_4bit"],
#             "gpu_memory_utilization": TRAIN_CONFIG["model"]["gpu_memory_utilization"],
#             "full_finetuning": TRAIN_CONFIG["model"]["full_finetuning"],
#         },
#         "lora": {
#             "r": TRAIN_CONFIG["lora"]["r"],
#             "lora_dropout": TRAIN_CONFIG["lora"]["lora_dropout"],
#             "target_modules": TRAIN_CONFIG["lora"]["target_modules"],
#             "bias": TRAIN_CONFIG["lora"]["bias"],
#         },
#         "training": {
#             "learning_rate": TRAIN_CONFIG["training"]["learning_rate"],
#             "adam_beta1": TRAIN_CONFIG["training"]["adam_beta1"],
#             "adam_beta2": TRAIN_CONFIG["training"]["adam_beta2"],
#             "weight_decay": TRAIN_CONFIG["training"]["weight_decay"],
#             "warmup_ratio": TRAIN_CONFIG["training"]["warmup_ratio"],
#             "lr_scheduler_type": TRAIN_CONFIG["training"]["lr_scheduler_type"],
#             "optim": TRAIN_CONFIG["training"]["optim"],
#             "per_device_train_batch_size": TRAIN_CONFIG["training"]["per_device_train_batch_size"],
#             "gradient_accumulation_steps": TRAIN_CONFIG["training"]["gradient_accumulation_steps"],
#             "num_generations": TRAIN_CONFIG["training"]["num_generations"],
#             "max_steps": TRAIN_CONFIG["training"]["max_steps"],
#             "save_steps": TRAIN_CONFIG["training"]["save_steps"],
#             "logging_steps": TRAIN_CONFIG["training"]["logging_steps"],
#             "max_grad_norm": TRAIN_CONFIG["training"]["max_grad_norm"],
#             "seed": TRAIN_CONFIG["training"]["seed"],
#         },
#         "rl": {
#             "temperature": TRAIN_CONFIG["grpo"]["temperature"],
#             "epsilon": TRAIN_CONFIG["grpo"]["epsilon"],
#             "epsilon_high": TRAIN_CONFIG["grpo"].get("epsilon_high"),
#             "beta": TRAIN_CONFIG["grpo"]["kl_coefficient"],
#             "loss_type": TRAIN_CONFIG["grpo"]["loss_type"],
#             "mask_truncated_completions": TRAIN_CONFIG["grpo"]["mask_truncated_completions"],
#         },
#         "gtpo": dict(TRAIN_CONFIG.get("gtpo", {})),
#         "mmr": dict(TRAIN_CONFIG.get("mmr", {})),
#         "data": {
#             "max_prompt_length": TRAIN_CONFIG["data"]["max_prompt_length"],
#             "max_completion_length": TRAIN_CONFIG["data"]["max_completion_length"],
#         },
#         "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
#     }
#     _grpo_out = Path(output_dir)
#     _grpo_out.mkdir(parents=True, exist_ok=True)
#     with open(_grpo_out / "training_hyperparameters.json", "w") as f:
#         json.dump(_grpo_hp, f, indent=2, ensure_ascii=False)

#     print(f"Model saved to {output_dir}")
#     !zip -r {output_dir}.zip {output_dir}
#     from IPython.display import FileLink
#     FileLink(f"{output_dir}.zip")
# else:
#     print(f"Skipping Stage 2 (MODE={MODE}).")
```

```python _cell_guid="a340693a-2ed6-4004-8dc6-e2167932d0ba" _uuid="baa6fd6c-06e6-48c0-a1e4-a75f9cdaf855" jupyter={"outputs_hidden": false}
# Free training model to avoid vLLM duplicate-layer crash
import gc
import os

# First: explicitly shutdown the vLLM engine before deleting model
if 'trainer' in dir() and trainer is not None:
    try:
        # Put vLLM to sleep and release GPU memory
        trainer.llm.sleep(sleep=True)
    except Exception:
        pass  # trainer may not have llm attribute in all modes
    del trainer

if 'model' in dir():
    del model

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Reset allocator to clear stale graph pools
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
```

```python _cell_guid="bb828e1c-7bfb-4763-8a6f-e165a0170052" _uuid="8869f1ee-d891-42b8-8898-7a6368ad29a0" jupyter={"outputs_hidden": false}
# Colab eval: download model from Hugging Face Hub
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

```python _cell_guid="ckpt-list-001" _uuid="ckpt-list-001" jupyter={"outputs_hidden": false}
# Checkpoint paths for multi-model evaluation
import zipfile, shutil

# Auto-select base path based on environment
if ENV_NAME == "kaggle":
    CKPT_BASE = "/kaggle/input/models/dzung271828/unsloth/transformers/default/6"
else:
    CKPT_BASE = "outputs/train_ckpts"

CKPT_LIST = {
    "sft_model_checkpoint-1335":                        f"{CKPT_BASE}/sft_model/sft_model",
    # "sft_grpo_model":                                   f"{CKPT_BASE}/sft_grpo_model",
    # "sft_rcgrpo_model":                                 f"{CKPT_BASE}/sft_rcgrpo_model",
    # "rctp_ft_model_checkpoint-1239":                    f"{CKPT_BASE}/rctp_ft_model/rctp_ft_model",
    # "rctp_ft_grpo_model":                               f"{CKPT_BASE}/rctp_ft_grpo_model",
    # "rctp_ft_rcgrpo_model":                             f"{CKPT_BASE}/rctp_ft_rcgrpo_model",
    # "rctp_ft_rcgrpo_KL0_numgen8":                       f"{CKPT_BASE}/rctp_ft_rcgrpo_KL0_numgen8",
    # "rctp_ft_rcgrpo_KL0_numgen16":                      f"{CKPT_BASE}/rctp_ft_rcgrpo_KL0_numgen16",
    # "rctp_ft_rcgrpo_KL0_numgen16_dapo":                 f"{CKPT_BASE}/rctp_ft_rcgrpo_KL0_numgen16_dapo",
    # "rctp_ft_rcgrpo_KL0_numgen16_cispo":                f"{CKPT_BASE}/rctp_ft_rcgrpo_KL0_numgen16_cispo",
    # "rctp_ft_rcgrpo_KL0_numgen16_cispo_highrewardprob0.2": f"{CKPT_BASE}/rctp_ft_rcgrpo_KL0_numgen16_cispo_highrewardprob0.2",
    # "rctp_ft_rcgrpo_KL0_numgen16_cispo_highrewardprob0.8": f"{CKPT_BASE}/rctp_ft_rcgrpo_KL0_numgen16_cispo_highrewardprob0.8",
}

print(f"Registered {len(CKPT_LIST)} checkpoints for evaluation:")
for name, path in CKPT_LIST.items():
    exists = "✓" if Path(path).exists() else "✗ MISSING"
    print(f"  {exists}  {name}: {path}")
```

```python _cell_guid="sanity-check-001" _uuid="sanity-check-001" jupyter={"outputs_hidden": false}
# ===================== SANITY CHECK: sample inference =====================
# Pick first available checkpoint and run inference on a few test samples.

# Find the first checkpoint that exists on disk
sanity_ckpt_name = None
sanity_ckpt_path = None
for name, path in CKPT_LIST.items():
    if Path(path).exists():
        sanity_ckpt_name = name
        sanity_ckpt_path = path
        break

if sanity_ckpt_path is None:
    print("ERROR: No checkpoint paths in CKPT_LIST exist on disk.")
else:
    print(f"Checkpoint: {sanity_ckpt_name}")
    print(f"Path:       {sanity_ckpt_path}")

    # Load model + tokenizer
    sanity_model, sanity_tokenizer = load_model(
        base_model_name=TRAIN_CONFIG["model"]["name"],
        adapter_model_path=sanity_ckpt_path,
        mode="inference",
        load_in_4bit=False,
        fast_inference=True,
        gpu_memory_utilization=TRAIN_CONFIG["model"]["gpu_memory_utilization"],
        env_name=ENV_NAME,
    )

    sanity_test_path = TRAIN_CONFIG["data"]["test_path"]
    import jsonlines
    sanity_samples = []
    with jsonlines.open(sanity_test_path) as f:
        for obj in f:
            sanity_samples.append(obj)

    print(f"\nTest dataset: {sanity_test_path} ({len(sanity_samples)} samples)")
    is_rc = "rcgrpo" in sanity_ckpt_name

    # Build prompts for first 3 samples
    _sanity_retriever = FunctionRetriever(function_library, method="hybrid")
    from vllm import SamplingParams as _SP
    sanity_prompts = []
    sanity_metas = []
    n_show = min(3, len(sanity_samples))

    for idx in range(n_show):
        sample = sanity_samples[idx]
        query = sample["query"]
        gt = sample.get("ground_truth", {})

        retrieved = _sanity_retriever.retrieve(query, k=10)

        if argument_values_catalog is not None:
            val_retriever = ArgumentValueRetriever(argument_values_catalog)
            arg_vals = val_retriever.retrieve_for_functions(query, retrieved, function_library)
        else:
            arg_vals = sample.get("retrieved_argument_values")

        messages = build_messages_for_grpo(query, retrieved, function_library, arg_vals)
        if is_rc and HIGH_REWARD_TOKEN in sanity_tokenizer.all_special_tokens:
            messages = inject_reward_token_into_messages(messages, HIGH_REWARD_TOKEN)

        prompt = sanity_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sanity_prompts.append(prompt)
        sanity_metas.append({"query": query, "gt": gt})

    # Generate with vLLM
    sp = _SP(temperature=0.0, max_tokens=2048, stop=["</tool_call>"], include_stop_str_in_output=True, seed=3407)
    outputs = sanity_model.fast_generate(sanity_prompts, sp)

    # Print results
    for idx, (output, meta) in enumerate(zip(outputs, sanity_metas)):
        response = output.outputs[0].text
        gt_calls = meta["gt"].get("calls", [])
        query = meta["query"]

        print(f"\n{'='*80}")
        print(f"  SAMPLE {idx + 1}")
        print(f"{'='*80}")
        print(f"  Query:   {query[:200]}{'...' if len(query) > 200 else ''}")
        print(f"  Reward:  {'HIGH' if is_rc else 'OFF'}")
        print(f"  Expected: {json.dumps(gt_calls, indent=4, ensure_ascii=False) if gt_calls else 'Abstain (no call)'}")
        print(f"  Response:\n{response}")
        print(f"{'='*80}")

    # Cleanup
    try:
        if hasattr(sanity_model, 'llm') and sanity_model.llm is not None:
            llm = sanity_model.llm
            if hasattr(llm, 'shutdown'):
                llm.shutdown()
    except Exception:
        pass
    try:
        del sanity_model, sanity_tokenizer
    except Exception:
        pass
    import gc as _gc2
    for _ in range(3):
        _gc2.collect()
    torch.cuda.empty_cache()
    print("\nModel unloaded. GPU memory released.")
```

```python _cell_guid="eval-config-001" _uuid="eval-config-001" jupyter={"outputs_hidden": false}
# ===================== EVAL CONFIG =====================
EVAL_CONFIG = {
    "top_k": 10,
    "max_new_tokens": 2048,
    "batch_size": 64,
    "gpu_memory_utilization": 0.8,
    "load_in_4bit": False,
    "use_vllm": True,
    "test_path": str(DATA_DIR / "test_dataset_cleaned.jsonl"),
}
print("Eval config set")
```

```python _cell_guid="eval-loop-001" _uuid="eval-loop-001" jupyter={"outputs_hidden": false}
# ===================== MULTI-MODEL EVALUATION LOOP =====================
import zipfile
import shutil

os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"

test_dataset_path = EVAL_CONFIG["test_path"]
REPORTS_DIR = Path("outputs/evaluation_reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _create_checkpoint_zip(checkpoint_name: str, report_dir: Path) -> Path:
    """Zip metrics_summary.csv + full_results.json into {checkpoint_name}.zip."""
    zip_path = REPORTS_DIR / f"{checkpoint_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["metrics_summary.csv", "full_results.json"]:
            fpath = report_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
    return zip_path

# Guard: no checkpoints registered
if len(CKPT_LIST) == 0:
    print("ERROR: CKPT_LIST is empty. Define checkpoint paths before running evaluation.")

elif not Path(test_dataset_path).exists():
    print(f"ERROR: Test dataset not found at {test_dataset_path}")

else:
    # Shared objects (created once, reused across checkpoints)
    retriever = FunctionRetriever(function_library, method="hybrid")
    sandbox = Sandbox(function_library)
    all_results = []

    for ckpt_name, ckpt_path in CKPT_LIST.items():
        ckpt_path_obj = Path(ckpt_path)
        if not ckpt_path_obj.exists():
            print(f"\n  SKIPPING {ckpt_name}: path {ckpt_path} does not exist")
            continue

        print(f"\n{'='*70}")
        print(f"  EVALUATING: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"{'='*70}")

        # Determine reward conditioning from checkpoint name convention
        is_rc_grpo = "rcgrpo" in ckpt_name

        # Pre-evaluation config summary
        print(f"  Config:")
        print(f"    Condition on high reward: {is_rc_grpo}")
        print(f"    Use vLLM:                {EVAL_CONFIG['use_vllm']}")
        print(f"    Batch size:              {EVAL_CONFIG['batch_size']}")
        print(f"    GPU mem utilization:     {EVAL_CONFIG['gpu_memory_utilization']}")
        print(f"    top_k:                   {EVAL_CONFIG['top_k']}")
        print(f"    max_new_tokens:          {EVAL_CONFIG['max_new_tokens']}")
        print(f"    load_in_4bit:            {EVAL_CONFIG['load_in_4bit']}")
        print(f"    Test dataset:            {test_dataset_path}")

        try:
            eval_result = evaluate_model(
                model_path=str(ckpt_path_obj),
                test_dataset_path=test_dataset_path,
                function_library=function_library,
                retriever=retriever,
                sandbox=sandbox,
                top_k=EVAL_CONFIG["top_k"],
                max_new_tokens=EVAL_CONFIG["max_new_tokens"],
                model_name_tag=ckpt_name,
                condition_on_high_reward=is_rc_grpo,
                argument_values=argument_values_catalog,
                use_vllm=EVAL_CONFIG["use_vllm"],
                batch_size=EVAL_CONFIG["batch_size"],
                gpu_memory_utilization=EVAL_CONFIG["gpu_memory_utilization"],
            )

            # Per-checkpoint report
            ckpt_report_dir = REPORTS_DIR / ckpt_name
            ckpt_report_dir.mkdir(parents=True, exist_ok=True)

            generate_report([eval_result], output_dir=str(ckpt_report_dir))

            # Zip the per-checkpoint results
            zip_path = _create_checkpoint_zip(ckpt_name, ckpt_report_dir)
            print(f"  Checkpoint results zipped -> {zip_path}")

            all_results.append(eval_result)

        except Exception as e:
            print(f"  FAILED {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup between checkpoints
            cleanup_training(trainer=None, model=None, tokenizer=None)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
            print(f"  Cleanup complete for {ckpt_name}")

    # Combined summary across all checkpoints
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("  COMBINED REPORT — All Checkpoints")
        print(f"{'='*70}")
        generate_report(all_results)
    elif len(all_results) == 1:
        print(f"\nSingle model evaluated. Full report at {REPORTS_DIR / list(CKPT_LIST.keys())[0]}")
        from IPython.display import FileLink
        FileLink(f"{REPORTS_DIR / list(CKPT_LIST.keys())[0]}.zip")

    print(f"\nEvaluation loop complete. Reports in {REPORTS_DIR}")
```
