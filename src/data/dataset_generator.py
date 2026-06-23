"""
dataset_generator.py
─────────────────────
Generates 2 000–2 500 (query, ground_truth_call) pairs by prompting
an LLM API (OpenAI / Anthropic / Google / Together / Mistral).

The generator reads two schema files:
  - function_schema_train.json  → functions used for training samples
  - function_schema_test.json   → functions held out for testing

and produces train_dataset.jsonl + test_dataset.jsonl.

Workflow distribution (configurable, defaults from project spec):
  60%  single_call   – one function call answers the query
  20%  parallel      – multiple independent calls in one turn
  15%  sequential    – chained calls where output feeds next input
   5%  abstention    – model should refuse / ask for more info

Train / Test split: functions are pre‑split by the schema files.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from tqdm.auto import tqdm
import logging

_log = logging.getLogger("tenacity.retry")
_log.setLevel(logging.DEBUG)
logging.basicConfig()  # ensure a handler exists


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class GroundTruth:
    function: str
    arguments: dict[str, Any]
    workflow_type: str  # single_call | parallel | sequential | abstention
    # For multi-call workflows
    calls: list[dict] = field(default_factory=list)


@dataclass
class DataSample:
    id: str
    query: str
    workflow_type: str
    function_name: str  # primary function (or "none" for abstention)
    ground_truth: dict  # serialised GroundTruth
    retrieved_functions: list[str]  # simulated top-k from library
    split: str = "train"  # train | test


# ──────────────────────────────────────────────────────────────────────────────
# API client factory
# ──────────────────────────────────────────────────────────────────────────────


class _APIClient:
    """
    Thin wrapper that normalises calls across providers.
    OpenRouter is treated as its own provider (not just an OpenAI base_url swap)
    because it requires extra headers that the plain OpenAI client does not send.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self._client = self._build_client(api_key, base_url)

    # ── Client factory ────────────────────────────────────────────────────────

    def _build_client(self, api_key: str, base_url: str | None):
        if self.provider == "openai":
            from openai import OpenAI

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return OpenAI(**kwargs)

        elif self.provider == "openrouter":
            from openai import OpenAI

            # OpenRouter requires HTTP-Referer + X-OpenRouter-Title on every request.
            # Pass them via default_headers so they appear on all calls
            # without any change to the call sites.
            return OpenAI(
                api_key=api_key,
                base_url=base_url or "https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/telco-agent-rl",
                    "X-OpenRouter-Title": "Telco Agent RL",
                },
            )

        elif self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=api_key)

        elif self.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model)

        elif self.provider == "together":
            from openai import OpenAI

            return OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.together.xyz/v1",
            )

        elif self.provider == "mistral":
            from mistralai import Mistral

            return Mistral(api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ── Completion call with verbose retry logging ────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,  # ← exposes real exception instead of RetryError wrapper
    )
    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        max_tokens: int = 1024,
    ) -> str:
        """Single chat completion — returns assistant text."""
        try:
            if self.provider in ("openai", "together", "openrouter"):
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            elif self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                )
                return resp.content[0].text

            elif self.provider == "google":
                resp = self._client.generate_content(f"{system}\n\n{user}")
                return resp.text

            elif self.provider == "mistral":
                resp = self._client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            raise RuntimeError(f"Unreachable provider branch: {self.provider}")

        except Exception as exc:
            # ── Print the real error before tenacity decides to retry ─────────
            # This is the line that was missing — without it, RetryError hides
            # the actual HTTP status, body, and message.
            print(
                f"\n  [DEBUG] {self.provider} API error "
                f"(model={self.model}): {type(exc).__name__}: {exc}"
            )
            if hasattr(exc, "status_code"):
                print(f"  [DEBUG] HTTP status : {exc.status_code}")
            if hasattr(exc, "body"):
                print(f"  [DEBUG] Error body  : {exc.body}")
            raise  # let tenacity handle retry / reraise


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a telecom network operations expert and dataset creator.
Your job is to generate realistic, diverse training samples for fine-tuning
a small language model on telecom function-calling tasks.

Always respond with **valid JSON only** — no markdown, no extra text, the entire query should be in Vietnamese.
Ensure the KPI code, location code, object code and unit code are based on these values:
kpi_code,Ý nghĩa
call_setup_success_rate,Tỷ lệ thiết lập cuộc gọi thành công
call_drop_rate,Tỷ lệ rớt cuộc gọi
data_traffic_volume,Lưu lượng dữ liệu
network_latency_avg_ms,Độ trễ mạng trung bình (ms)
service_cancellation_rate,Tỷ lệ hủy dịch vụ
voice_call_duration_avg,Thời lượng cuộc gọi thoại trung bình (phút)
data_usage_per_user_4g,Mức sử dụng dữ liệu trung bình trên mỗi thuê bao 4G
active_user_count_4g,Số lượng người dùng 4G hoạt động
interruption_incident_count,Số lượng sự cố gián đoạn dịch vụ
subscriber_growth_rate,Tỷ lệ tăng trưởng thuê bao mới
coverage_percentage_4g,Tỷ lệ phủ sóng 4G (%)
user_experience_score,Điểm trải nghiệm người dùng

Nhóm,Giá trị khả dụng,Mã
Toàn quốc,Việt Nam,VNM
Toàn quốc,toàn quốc,VNM
Khu vực,Khu vực 1,KV1
Khu vực,Khu vực miền Bắc,KV1
Khu vực,Khu vực 2,KV2
Khu vực,Khu vực miền Trung,KV2
Khu vực,Khu vực 3,KV3
Khu vực,Khu vực miền Nam,KV3
Tỉnh/Thành phố Việt Nam,Hà Nội,HNI
Tỉnh/Thành phố Việt Nam,Huế,HUE
Tỉnh/Thành phố Việt Nam,Lai Châu,LCU
Tỉnh/Thành phố Việt Nam,Điện Biên,DBN
Tỉnh/Thành phố Việt Nam,Sơn La,SLA
Tỉnh/Thành phố Việt Nam,Lạng Sơn,LSN
Tỉnh/Thành phố Việt Nam,Quảng Ninh,QNH
Tỉnh/Thành phố Việt Nam,Thanh Hóa,THA
Tỉnh/Thành phố Việt Nam,Nghệ An,NAN
Tỉnh/Thành phố Việt Nam,Hà Tĩnh,HTH
Tỉnh/Thành phố Việt Nam,Cao Bằng,CBG
Tỉnh/Thành phố Việt Nam,Tuyên Quang,TQG
Tỉnh/Thành phố Việt Nam,Lào Cai,LCI
Tỉnh/Thành phố Việt Nam,Thái Nguyên,TNN
Tỉnh/Thành phố Việt Nam,Phú Thọ,PTO
Tỉnh/Thành phố Việt Nam,Bắc Ninh,BNH
Tỉnh/Thành phố Việt Nam,Hưng Yên,HYN
Tỉnh/Thành phố Việt Nam,Hải Phòng,HPG
Tỉnh/Thành phố Việt Nam,Ninh Bình,NBH
Tỉnh/Thành phố Việt Nam,Quảng Trị,QTI
Tỉnh/Thành phố Việt Nam,Đà Nẵng,DNG
Tỉnh/Thành phố Việt Nam,Quảng Ngãi,QNI
Tỉnh/Thành phố Việt Nam,Gia Lai,GLI
Tỉnh/Thành phố Việt Nam,Khánh Hòa,KHA
Tỉnh/Thành phố Việt Nam,Lâm Đồng,LDG
Tỉnh/Thành phố Việt Nam,Đắk Lắk,DLK
Tỉnh/Thành phố Việt Nam,Thành phố Hồ Chí Minh,HCM
Tỉnh/Thành phố Việt Nam,Đồng Nai,DNI
Tỉnh/Thành phố Việt Nam,Tây Ninh,TNH
Tỉnh/Thành phố Việt Nam,Cần Thơ,CTO
Tỉnh/Thành phố Việt Nam,Vĩnh Long,VLG
Tỉnh/Thành phố Việt Nam,Đồng Tháp,DTP
Tỉnh/Thành phố Việt Nam,Cà Mau,CMU
Tỉnh/Thành phố Việt Nam,An Giang,AGG
Quốc gia,Cambodia,KHM
Quốc gia,Laos,LAO
Quốc gia,Peru,PER
Quốc gia,Myanmar,MMR
Quốc gia,TimorLeste,TLS
Quốc gia,Haiti,HTI
Quốc gia,Mozambique,MOZ
Quốc gia,Tanzania,TZA
Quốc gia,Cameroon,CMR
Quốc gia,Burundi,BDI

unit_code,Đơn vị/Công ty
VCS,Công ty An ninh mạng Viettel
IDC,Công ty TNHH Viettel - IDC
VTM,Công ty Truyền thông Viettel
VAI,Trung tâm Trí tuệ nhân tạo và Dịch vụ dữ liệu Việt Nam
VTPost,Tổng công ty cổ phần Bưu chính Viettel
VDS,Tổng công ty Dịch vụ số Viettel
VTS,Tổng công ty Giải pháp doanh nghiệp Viettel
VTNet,Tổng công ty Mạng lưới Viettel
VTT,Tổng Công ty Viễn thông Viettel
"""

_SINGLE_CALL_TEMPLATE = """\
Given the following telecom network function:

FUNCTION SCHEMA:
{schema}

Generate {n} diverse, realistic natural-language queries that a network
operations engineer would ask, along with the exact function call
(with realistic argument values) that answers each query.

Respond with a JSON array of exactly {n} objects, each with:
{{
  "query": "<natural language question>",
  "reasoning": "<brief step-by-step reasoning>",
  "function_call": {{
    "function": "<function_name>",
    "arguments": {{ <param>: <value>, ... }}
  }}
}}

Rules:
- Vary phrasing, urgency, and technical detail across queries
- Use realistic telecom values (cell IDs, frequencies, thresholds, etc.)
- All required parameters must be present
- Respect any constraints in the schema
- Do NOT invent parameters not in the schema
"""

_PARALLEL_CALL_TEMPLATE = """\
Given these telecom network functions:

FUNCTION SCHEMAS:
{schemas}

Generate {n} queries that require calling MULTIPLE of these functions
simultaneously (in parallel) to fully answer the question.

Respond with a JSON array of exactly {n} objects:
{{
  "query": "<natural language question requiring multiple tools>",
  "reasoning": "<why multiple calls are needed>",
  "function_calls": [
    {{"function": "<name1>", "arguments": {{ ... }}}},
    {{"function": "<name2>", "arguments": {{ ... }}}}
  ]
}}
"""

_SEQUENTIAL_CALL_TEMPLATE = """\
Given these telecom network functions:

FUNCTION SCHEMAS:
{schemas}

Generate {n} queries that require chained/sequential function calls
(the output of one call informs the next).

Respond with a JSON array of exactly {n} objects:
{{
  "query": "<query needing sequential tool calls>",
  "reasoning": "<chain-of-thought for the sequence>",
  "function_calls": [
    {{"step": 1, "function": "<name1>", "arguments": {{ ... }}, "depends_on": []}},
    {{"step": 2, "function": "<name2>", "arguments": {{ ... }}, "depends_on": [1]}}
  ]
}}
"""

_ABSTENTION_TEMPLATE = """\
Given these telecom network functions:

FUNCTION SCHEMAS:
{schemas}

Generate {n} queries where the model should NOT call any function because:
- Required information is missing from the query, OR
- The query is out of scope for these functions, OR
- The user needs to provide clarification first.

Respond with a JSON array of exactly {n} objects:
{{
  "query": "<ambiguous or incomplete query>",
  "reasoning": "<why no function should be called>",
  "function_call": null,
  "refusal_message": "<what the model should say instead>"
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Core generator
# ──────────────────────────────────────────────────────────────────────────────


class TelcoDatasetGenerator:
    """
    Generates GRPO‑ready training samples from separate train/test schema files.

    Usage
    ─────
    gen = TelcoDatasetGenerator.from_schemas(
        train_schema_path="data/processed/function_schema_train.json",
        test_schema_path="data/processed/function_schema_test.json",
        provider="openai",
        model="gpt-4o-mini",
    )
    train, test = gen.generate(total=2400, output_dir="data/processed")
    """

    def __init__(
        self,
        train_function_library: dict,
        test_function_library: dict,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_workers: int = 8,
        requests_per_minute: int = 500,
        temperature: float = 0.9,
        max_tokens: int = 1024,
        seed: int = 42,
    ):
        self.train_library = train_function_library
        self.test_library = test_function_library
        self.train_func_names = list(train_function_library.keys())
        self.test_func_names = list(test_function_library.keys())
        # Union for retrieval simulation
        self.all_func_names = list(set(self.train_func_names + self.test_func_names))
        self.max_workers = max_workers
        self.rpm = requests_per_minute
        self.temperature = temperature
        self.max_tokens = max_tokens
        random.seed(seed)

        _key = (
            api_key
            or os.getenv(f"{provider.upper()}_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        if not _key:
            raise ValueError(
                f"API key not found. Set the environment variable "
                f"'{provider.upper()}_API_KEY' or pass api_key= explicitly."
            )
        self.client = _APIClient(provider, model, _key, base_url)
        print(
            f"[DatasetGenerator] Provider={provider}  Model={model}  "
            f"Train functions={len(self.train_func_names)}  "
            f"Test functions={len(self.test_func_names)}  "
            f"MaxWorkers={max_workers}  RPM={requests_per_minute}"
        )

    # ── factory methods ──────────────────────────────────────────────────────

    @classmethod
    def from_schemas(
        cls,
        train_schema_path: str,
        test_schema_path: str,
        **kwargs,
    ) -> "TelcoDatasetGenerator":
        """Build generator from two separate schema JSON files."""
        with open(train_schema_path, "r", encoding="utf-8") as fh:
            train_library = json.load(fh)
        with open(test_schema_path, "r", encoding="utf-8") as fh:
            test_library = json.load(fh)
        return cls(
            train_function_library=train_library,
            test_function_library=test_library,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: dict) -> "TelcoDatasetGenerator":
        """Build from a loaded YAML config dict (dataset_generation section)."""
        dg = config.get("dataset_generation", config)
        train_schema = config.get("data", {}).get(
            "train_schema_path", "data/processed/function_schema_train.json"
        )
        test_schema = config.get("data", {}).get(
            "test_schema_path", "data/processed/function_schema_test.json"
        )
        return cls.from_schemas(
            train_schema_path=train_schema,
            test_schema_path=test_schema,
            provider=dg.get("provider", "openai"),
            model=dg.get("model", "gpt-4o-mini"),
            api_key=os.getenv(dg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=dg.get("base_url"),
            max_workers=dg.get("max_workers", 8),
            requests_per_minute=dg.get("requests_per_minute", 500),
            temperature=dg.get("temperature", 0.9),
            max_tokens=dg.get("max_tokens", 1024),
        )

    # ── public entry-point ────────────────────────────────────────────────────

    def generate(
        self,
        total: int = 2400,
        output_dir: str = "data/processed",
        workflow_distribution: dict | None = None,
        train_split: float = 0.89,
    ) -> tuple[list[DataSample], list[DataSample]]:
        """
        Main generation loop.

        Parameters
        ──────────
        total                   : target number of samples
        output_dir              : where to write .jsonl files
        workflow_distribution   : overrides default ratios
        train_split             : fraction of samples for training (within training functions)

        Returns
        ───────
        (train_samples, test_samples)
        """
        dist = workflow_distribution or {
            "single_call": 0.60,
            "parallel": 0.20,
            "sequential": 0.15,
            "abstention": 0.05,
        }

        train_funcs = self.train_func_names
        test_funcs = self.test_func_names

        print(f"[DatasetGenerator] Train functions : {len(train_funcs)}")
        print(f"[DatasetGenerator] Test  functions : {test_funcs}")

        # ── 2. Count per workflow type ───────────────────────────────────────
        counts = {wt: max(1, int(total * ratio)) for wt, ratio in dist.items()}
        # adjust to hit exact total
        diff = total - sum(counts.values())
        counts["single_call"] += diff

        print(f"[DatasetGenerator] Workflow counts : {counts}")

        # ── 3. Generate samples per workflow ────────────────────────────────
        all_samples: list[DataSample] = []
        all_samples += self._generate_single_calls(
            counts["single_call"], train_funcs, split="train"
        )
        all_samples += self._generate_parallel(
            counts["parallel"], train_funcs, split="train"
        )
        all_samples += self._generate_sequential(
            counts["sequential"], train_funcs, split="train"
        )
        all_samples += self._generate_abstentions(
            counts["abstention"], train_funcs, split="train"
        )

        # ── 4. Generate test samples from held-out functions ─────────────────
        test_count = max(50, int(total * (1 - train_split)))
        test_samples = self._generate_single_calls(test_count, test_funcs, split="test")

        # ── 5. Assign splits + simulate retrieved functions ──────────────────
        random.shuffle(all_samples)
        train_cut = int(len(all_samples) * train_split)
        train_samples = all_samples[:train_cut]
        # remaining training‑pool samples go to test too (edge cases / overflow)
        extra_test = all_samples[train_cut:]
        test_samples = test_samples + extra_test

        for s in train_samples:
            s.split = "train"
        for s in test_samples:
            s.split = "test"

        self._simulate_retrieval(train_samples, k=5)
        self._simulate_retrieval(test_samples, k=5)

        # ── 6. Save ──────────────────────────────────────────────────────────
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._save_jsonl(train_samples, out / "raw_train_dataset.jsonl")
        self._save_jsonl(test_samples, out / "raw_test_dataset.jsonl")

        print(
            f"\n[DatasetGenerator] ✓  train={len(train_samples)}  test={len(test_samples)}"
        )
        return train_samples, test_samples

    # ── workflow generators ───────────────────────────────────────────────────

    def _generate_single_calls(
        self,
        count: int,
        func_pool: list[str],
        split: str = "train",
        batch_size: int = 5,
    ) -> list[DataSample]:
        """Generate single-function-call samples."""
        tasks: list[tuple[str, int]] = []  # (func_name, n_per_call)
        remaining = count
        while remaining > 0:
            fn = random.choice(func_pool)
            n = min(batch_size, remaining)
            tasks.append((fn, n))
            remaining -= n

        samples: list[DataSample] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_single, fn, n): (fn, n) for fn, n in tasks
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="single_call",
                leave=False,
            ):
                fn, _ = futures[fut]
                try:
                    raw_list = fut.result()
                    for raw in raw_list:
                        s = self._parse_single(raw, fn, split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    print(f"  [WARN] single_call error for '{fn}': {exc}")

        self._rate_limit_wait(len(tasks))
        return samples

    def _generate_parallel(
        self, count: int, func_pool: list[str], split: str = "train"
    ) -> list[DataSample]:
        """Generate parallel multi-call samples."""
        tasks: list[tuple[list[str], int]] = []
        remaining = count
        while remaining > 0:
            n_funcs = random.randint(2, min(4, len(func_pool)))
            chosen = random.sample(func_pool, n_funcs)
            n = min(3, remaining)
            tasks.append((chosen, n))
            remaining -= n

        samples: list[DataSample] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_parallel, fns, n): fns for fns, n in tasks
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="parallel", leave=False
            ):
                fns = futures[fut]
                try:
                    for raw in fut.result():
                        s = self._parse_parallel(raw, fns, split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    print(f"  [WARN] parallel error: {exc}")

        return samples

    def _generate_sequential(
        self, count: int, func_pool: list[str], split: str = "train"
    ) -> list[DataSample]:
        """Generate sequential / chained call samples."""
        tasks: list[tuple[list[str], int]] = []
        remaining = count
        while remaining > 0:
            n_funcs = random.randint(2, min(3, len(func_pool)))
            chosen = random.sample(func_pool, n_funcs)
            n = min(3, remaining)
            tasks.append((chosen, n))
            remaining -= n

        samples: list[DataSample] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_sequential, fns, n): fns for fns, n in tasks
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="sequential",
                leave=False,
            ):
                fns = futures[fut]
                try:
                    for raw in fut.result():
                        s = self._parse_sequential(raw, fns, split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    print(f"  [WARN] sequential error: {exc}")

        return samples

    def _generate_abstentions(
        self, count: int, func_pool: list[str], split: str = "train"
    ) -> list[DataSample]:
        """Generate refusal / abstention samples."""
        tasks: list[tuple[list[str], int]] = []
        remaining = count
        while remaining > 0:
            n_funcs = random.randint(1, min(3, len(func_pool)))
            chosen = random.sample(func_pool, n_funcs)
            n = min(3, remaining)
            tasks.append((chosen, n))
            remaining -= n

        samples: list[DataSample] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_abstention, fns, n): fns for fns, n in tasks
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="abstention",
                leave=False,
            ):
                fns = futures[fut]
                try:
                    for raw in fut.result():
                        s = self._parse_abstention(raw, fns, split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    print(f"  [WARN] abstention error: {exc}")

        return samples

    # ── API call methods ──────────────────────────────────────────────────────

    def _schema_str(self, func_name: str) -> str:
        """
        Return JSON schema for a given function, searching first in train_library,
        then test_library (for test‑only functions during retrieval simulation).
        """
        if func_name in self.train_library:
            schema = self.train_library[func_name]
        elif func_name in self.test_library:
            schema = self.test_library[func_name]
        else:
            raise KeyError(f"Function '{func_name}' not found in any library.")
        return json.dumps(schema, indent=2)

    def _schemas_str(self, func_names: list[str]) -> str:
        return "\n---\n".join(self._schema_str(fn) for fn in func_names)

    def _call_single(self, func_name: str, n: int) -> list[dict]:
        prompt = _SINGLE_CALL_TEMPLATE.format(
            schema=self._schema_str(func_name),
            n=n,
        )
        text = self.client.complete(
            SYSTEM_PROMPT, prompt, self.temperature, self.max_tokens
        )
        return self._parse_json_list(text)

    def _call_parallel(self, func_names: list[str], n: int) -> list[dict]:
        prompt = _PARALLEL_CALL_TEMPLATE.format(
            schemas=self._schemas_str(func_names),
            n=n,
        )
        text = self.client.complete(
            SYSTEM_PROMPT, prompt, self.temperature, self.max_tokens
        )
        return self._parse_json_list(text)

    def _call_sequential(self, func_names: list[str], n: int) -> list[dict]:
        prompt = _SEQUENTIAL_CALL_TEMPLATE.format(
            schemas=self._schemas_str(func_names),
            n=n,
        )
        text = self.client.complete(
            SYSTEM_PROMPT, prompt, self.temperature, self.max_tokens
        )
        return self._parse_json_list(text)

    def _call_abstention(self, func_names: list[str], n: int) -> list[dict]:
        prompt = _ABSTENTION_TEMPLATE.format(
            schemas=self._schemas_str(func_names),
            n=n,
        )
        text = self.client.complete(
            SYSTEM_PROMPT, prompt, self.temperature, self.max_tokens
        )
        return self._parse_json_list(text)

    # ── parsers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_list(text: str) -> list[dict]:
        """Extract the first JSON array from LLM output."""
        text = text.strip()
        # strip markdown fences if present
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            # try to find JSON array via bracket matching
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except Exception:
                    pass
            return []

    def _parse_single(self, raw: dict, func_name: str, split: str) -> DataSample | None:
        query = raw.get("query", "").strip()
        call = raw.get("function_call")
        if not query or not call:
            return None
        calls = [{"function": call.get("function", func_name), "arguments": call.get("arguments", {})}]
        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type="single_call",
            function_name=func_name,
            ground_truth={
                "calls": calls,
                "workflow": "single_call",
                "reasoning": raw.get("reasoning", ""),
            },
            retrieved_functions=[],
            split=split,
        )

    def _parse_parallel(
        self, raw: dict, func_names: list[str], split: str = "train"
    ) -> DataSample | None:
        query = raw.get("query", "").strip()
        raw_calls = raw.get("function_calls", [])
        if not query or not raw_calls:
            return None
        primary = raw_calls[0].get("function", func_names[0]) if raw_calls else func_names[0]
        calls = [
            {"function": c.get("function", primary), "arguments": c.get("arguments", {})}
            for c in raw_calls
        ]
        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type="parallel",
            function_name=primary,
            ground_truth={
                "calls": calls,
                "workflow": "parallel",
                "reasoning": raw.get("reasoning", ""),
            },
            retrieved_functions=[],
            split=split,
        )

    def _parse_sequential(
        self, raw: dict, func_names: list[str], split: str = "train"
    ) -> DataSample | None:
        query = raw.get("query", "").strip()
        raw_calls = raw.get("function_calls", [])
        if not query or not raw_calls:
            return None
        first_call = raw_calls[0] if raw_calls else {}
        calls = [
            {"function": c.get("function", func_names[0]), "arguments": c.get("arguments", {})}
            for c in raw_calls
        ]
        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type="sequential",
            function_name=first_call.get("function", func_names[0]),
            ground_truth={
                "calls": calls,
                "workflow": "sequential",
                "reasoning": raw.get("reasoning", ""),
            },
            retrieved_functions=[],
            split=split,
        )

    def _parse_abstention(
        self, raw: dict, func_names: list[str], split: str = "train"
    ) -> DataSample | None:
        query = raw.get("query", "").strip()
        if not query:
            return None
        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type="abstention",
            function_name="none",
            ground_truth={
                "calls": [],
                "workflow": "abstention",
                "refusal_message": raw.get("refusal_message", ""),
                "reasoning": raw.get("reasoning", ""),
            },
            retrieved_functions=[],
            split=split,
        )

    # ── retrieval simulation ──────────────────────────────────────────────────

    def _simulate_retrieval(self, samples: list[DataSample], k: int = 5) -> None:
        """
        Fill retrieved_functions with the ground-truth function + (k-1) distractors.
        This simulates what the real retriever will return at inference time.
        Real retrieval happens in src/data/retrieval.py at training time.
        """
        for s in samples:
            true_fn = s.function_name
            distractors = [fn for fn in self.all_func_names if fn != true_fn]
            chosen_distractors = random.sample(
                distractors, min(k - 1, len(distractors))
            )
            pool = (
                ([true_fn] + chosen_distractors)
                if true_fn != "none"
                else chosen_distractors
            )
            random.shuffle(pool)
            s.retrieved_functions = pool[:k]

    # ── I/O helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _save_jsonl(samples: list[DataSample], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        print(f"[DatasetGenerator] Saved {len(samples)} samples → {path}")

    def _rate_limit_wait(self, n_calls: int) -> None:
        """Naive rate-limit: sleep if we're generating too fast."""
        if self.rpm > 0:
            min_interval = n_calls / (self.rpm / 60)
            if min_interval > 0.5:
                time.sleep(min_interval)
