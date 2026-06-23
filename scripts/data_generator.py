"""
dataset_generator.py
─────────────────────
Generates 2 000–2 500 (query, ground_truth) pairs by prompting
an LLM API (OpenAI / Anthropic / Google / Together / Mistral).

The generator reads two schema files:
  - function_schema_train.json  → functions used for training samples
  - function_schema_test.json   → functions held out for testing

and produces train_dataset.jsonl + test_dataset.jsonl.

Workflow distribution (configurable, defaults):
  70%  single_call   – one function call answers the query
  20%  parallel      – multiple independent calls in one turn
  10%  abstention    – model should refuse / ask for more info

Output format (per line):
{
  "query": "...",
  "workflow_type": "single_call" | "parallel" | "abstention",
  "ground_truth": {
    "calls": [{"function": "...", "arguments": {...}}, ...],
    "reasoning": "...",
    "refusal_message": "..."   // only for abstention
  }
}
"""

from __future__ import annotations

import json
import logging
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
)
from tqdm.auto import tqdm

# ── Logging setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# Silence tenacity logs unless needed
logging.getLogger("tenacity.retry").setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataSample:
    id: str
    query: str
    workflow_type: str
    ground_truth: dict          # contains calls, reasoning, (optional) refusal_message
    retrieved_functions: list[str]  # simulated top‑k from library
    split: str = "train"


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
        logger.info(f"API client initialized: provider={provider}, model={model}")

    def _build_client(self, api_key: str, base_url: str | None):
        if self.provider == "openai":
            from openai import OpenAI
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return OpenAI(**kwargs)

        elif self.provider == "openrouter":
            from openai import OpenAI
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
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
            logger.warning(
                f"API error (provider={self.provider}, model={self.model}): "
                f"{type(exc).__name__}: {exc}"
            )
            if hasattr(exc, "status_code"):
                logger.warning(f"HTTP status: {exc.status_code}")
            if hasattr(exc, "body"):
                logger.warning(f"Error body: {exc.body}")
            raise


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

Respond with a JSON array of exactly {n} objects, each with this structure:
{{
  "query": "<natural language question in Vietnamese>",
  "workflow_type": "single_call",
  "ground_truth": {{
    "calls": [
      {{"function": "<function_name>", "arguments": {{ <param>: <value>, ... }}}}
    ],
    "reasoning": "<brief step‑by‑step reasoning in Vietnamese>"
  }}
}}

Rules:
- Vary phrasing, urgency, and technical detail across queries
- Use realistic telecom values (cell IDs, frequencies, thresholds, etc.)
- The function_name must match the function provided above
- All required parameters must be present
- Respect any constraints in the schema
- Do NOT invent parameters not in the schema
- The query must be in Vietnamese
- Return **only** the JSON array, no extra text.
"""

_PARALLEL_CALL_TEMPLATE = """\
Given these telecom network functions:

FUNCTION SCHEMAS:
{schemas}

Generate {n} queries that require calling MULTIPLE of these functions
simultaneously (in parallel) to fully answer the question.

Respond with a JSON array of exactly {n} objects, each with this structure:
{{
  "query": "<natural language question in Vietnamese requiring multiple tools>",
  "workflow_type": "parallel",
  "ground_truth": {{
    "calls": [
      {{"function": "<name1>", "arguments": {{ ... }}}},
      {{"function": "<name2>", "arguments": {{ ... }}}}
    ],
    "reasoning": "<why multiple calls are needed, in Vietnamese>"
  }}
}}

Rules:
- Vary phrasing and technical detail across queries
- Each call must match one of the provided functions
- All required parameters must be present in every call
- The query must be in Vietnamese
- Return **only** the JSON array, no extra text.
"""

_ABSTENTION_TEMPLATE = """\
Given these telecom network functions:

FUNCTION SCHEMAS:
{schemas}

Generate {n} queries where the model should NOT call any function because:
- Required information is missing from the query, OR
- The query is out of scope for these functions, OR
- The user needs to provide clarification first.

Respond with a JSON array of exactly {n} objects, each with this structure:
{{
  "query": "<ambiguous or incomplete query in Vietnamese>",
  "workflow_type": "abstention",
  "ground_truth": {{
    "calls": [],
    "refusal_message": "<helpful message in Vietnamese explaining what's missing>",
    "reasoning": "<why no function should be called, in Vietnamese>"
  }}
}}

Rules:
- The query must be ambiguous or incomplete so that no schema function can answer it
- The refusal_message should be helpful and in Vietnamese
- function_call must be null (never return a fake function name)
- Return **only** the JSON array, no extra text.
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
        logger.info(
            f"Generator initialized: provider={provider}, model={model}, "
            f"train_funcs={len(self.train_func_names)}, test_funcs={len(self.test_func_names)}, "
            f"max_workers={max_workers}, rpm={requests_per_minute}, seed={seed}"
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
        logger.info(f"Loading train schema from {train_schema_path}")
        with open(train_schema_path, "r", encoding="utf-8") as fh:
            train_library = json.load(fh)
        logger.info(f"Loading test schema from {test_schema_path}")
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
        logger.info(f"Starting dataset generation: total={total}, train_split={train_split}")
        dist = workflow_distribution or {
            "single_call": 0.70,
            "parallel": 0.20,
            "abstention": 0.10,
        }
        logger.info(f"Workflow distribution: {dist}")

        train_funcs = self.train_func_names
        test_funcs = self.test_func_names

        logger.info(f"Train functions available: {len(train_funcs)}")
        logger.info(f"Test functions available: {len(test_funcs)}")

        # ── 2. Count per workflow type ───────────────────────────────────────
        counts = {wt: max(1, int(total * ratio)) for wt, ratio in dist.items()}
        # adjust to hit exact total
        diff = total - sum(counts.values())
        counts["single_call"] += diff
        logger.info(f"Workflow sample counts: {counts}")

        # ── 3. Generate samples per workflow ────────────────────────────────
        all_samples: list[DataSample] = []
        logger.info("Generating training‑pool samples...")
        all_samples += self._generate_single_calls(
            counts["single_call"], train_funcs, split="train"
        )
        all_samples += self._generate_parallel(
            counts["parallel"], train_funcs, split="train"
        )
        all_samples += self._generate_abstentions(
            counts["abstention"], train_funcs, split="train"
        )

        # ── 4. Generate test samples from held‑out functions ─────────────────
        test_count = max(50, int(total * (1 - train_split)))
        logger.info(f"Generating {test_count} test samples from held‑out functions...")
        test_samples = self._generate_single_calls(test_count, test_funcs, split="test")

        # ── 5. Assign splits + simulate retrieved functions ──────────────────
        random.shuffle(all_samples)
        train_cut = int(len(all_samples) * train_split)
        train_samples = all_samples[:train_cut]
        extra_test = all_samples[train_cut:]
        test_samples = test_samples + extra_test

        for s in train_samples:
            s.split = "train"
        for s in test_samples:
            s.split = "test"

        logger.info(f"Simulating retrieval for train ({len(train_samples)}) and test ({len(test_samples)}) samples...")
        self._simulate_retrieval(train_samples, k=5)
        self._simulate_retrieval(test_samples, k=5)

        # ── 6. Save ──────────────────────────────────────────────────────────
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_path = out / "raw_train_dataset.jsonl"
        test_path = out / "raw_test_dataset.jsonl"
        self._save_jsonl(train_samples, train_path)
        self._save_jsonl(test_samples, test_path)

        logger.info(
            f"Generation complete: train={len(train_samples)}, test={len(test_samples)} "
            f"(saved to {out})"
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
        """Generate single‑function‑call samples."""
        logger.info(f"Generating {count} single_call samples (split={split})")
        tasks: list[tuple[str, int]] = []
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
                        s = self._parse_sample(raw, "single_call", split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    logger.warning(f"single_call error for '{fn}': {exc}")

        self._rate_limit_wait(len(tasks))
        logger.debug(f"Generated {len(samples)} single_call samples")
        return samples

    def _generate_parallel(
        self, count: int, func_pool: list[str], split: str = "train"
    ) -> list[DataSample]:
        """Generate parallel multi‑call samples."""
        logger.info(f"Generating {count} parallel samples (split={split})")
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
                        s = self._parse_sample(raw, "parallel", split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    logger.warning(f"parallel error: {exc}")

        self._rate_limit_wait(len(tasks))
        logger.debug(f"Generated {len(samples)} parallel samples")
        return samples

    def _generate_abstentions(
        self, count: int, func_pool: list[str], split: str = "train"
    ) -> list[DataSample]:
        """Generate refusal / abstention samples."""
        logger.info(f"Generating {count} abstention samples (split={split})")
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
                        s = self._parse_sample(raw, "abstention", split)
                        if s:
                            samples.append(s)
                except Exception as exc:
                    logger.warning(f"abstention error: {exc}")

        self._rate_limit_wait(len(tasks))
        logger.debug(f"Generated {len(samples)} abstention samples")
        return samples

    # ── API call methods ──────────────────────────────────────────────────────

    def _schema_str(self, func_name: str) -> str:
        """Return JSON schema for a given function."""
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
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            # fallback: try to extract array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except Exception:
                    pass
            return []

    def _parse_sample(self, raw: dict, expected_wf: str, split: str) -> DataSample | None:
        """Parse a raw dictionary into a DataSample, enforcing the required format."""
        query = raw.get("query", "").strip()
        workflow_type = raw.get("workflow_type", expected_wf)
        ground_truth = raw.get("ground_truth")

        if not query:
            logger.warning("Missing 'query' in raw sample")
            return None
        if not ground_truth or not isinstance(ground_truth, dict):
            logger.warning("Missing or invalid 'ground_truth' in raw sample")
            return None

        # Ensure ground_truth has 'calls' and 'reasoning'
        calls = ground_truth.get("calls", [])
        if not isinstance(calls, list):
            logger.warning("'calls' in ground_truth is not a list")
            return None

        # For abstention, calls must be empty; for others, non‑empty
        if workflow_type == "abstention":
            if calls:
                logger.warning("Abstention sample has non‑empty calls")
                return None
            if "refusal_message" not in ground_truth:
                logger.warning("Abstention sample missing 'refusal_message'")
                return None
        else:
            if not calls:
                logger.warning(f"{workflow_type} sample has empty calls")
                return None

        # Build the final ground_truth dict exactly as required
        gt_out = {
            "calls": calls,
            "reasoning": ground_truth.get("reasoning", ""),
        }
        if workflow_type == "abstention":
            gt_out["refusal_message"] = ground_truth.get("refusal_message", "")

        return DataSample(
            id=str(uuid.uuid4()),
            query=query,
            workflow_type=workflow_type,
            ground_truth=gt_out,
            retrieved_functions=[],
            split=split,
        )

    # ── retrieval simulation ──────────────────────────────────────────────────

    def _simulate_retrieval(self, samples: list[DataSample], k: int = 5) -> None:
        """Fill retrieved_functions with the ground‑truth function + distractors."""
        for s in samples:
            calls = s.ground_truth.get("calls", [])
            true_fn = calls[0]["function"] if calls else None
            if true_fn:
                distractors = [fn for fn in self.all_func_names if fn != true_fn]
                chosen = random.sample(distractors, min(k - 1, len(distractors)))
                pool = [true_fn] + chosen
            else:
                pool = []
            random.shuffle(pool)
            s.retrieved_functions = pool[:k]

    # ── I/O helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _save_jsonl(samples: list[DataSample], path: Path) -> None:
        """Save samples in the required schema (query, workflow_type, ground_truth)."""
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                out = {
                    "query": s.query,
                    "workflow_type": s.workflow_type,
                    "ground_truth": s.ground_truth,
                }
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(samples)} samples to {path}")

    def _rate_limit_wait(self, n_calls: int) -> None:
        """Naive rate‑limit: sleep if we're generating too fast."""
        if self.rpm > 0:
            min_interval = n_calls / (self.rpm / 60)
            if min_interval > 0.5:
                logger.debug(f"Rate‑limiting: sleeping {min_interval:.2f}s")
                time.sleep(min_interval)


# ──────────────────────────────────────────────────────────────────────────────
# Command‑line entry point (optional)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: generate dataset with default settings
    generator = TelcoDatasetGenerator.from_schemas(
        train_schema_path="data/processed/function_schema_train.json",
        test_schema_path="data/processed/function_schema_test.json",
        provider="openai",
        model="gpt-4o-mini",
    )
    generator.generate(total=2400, output_dir="data/processed")