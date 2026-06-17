import os, json
from openai import OpenAI

# ── 1. Key is set ──────────────────────────────────────────────────────────────
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is not set!"
print("✓ Key found:", os.environ["OPENROUTER_API_KEY"][:8], "...")

# ── 2. Key is valid ────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/telco-agent-rl",
        "X-Title": "Telco Agent RL",
    },
)

# ── 3. Model name is correct ──────────────────────────────────────────────────
MODEL = "meta-llama/llama-3.3-70b-instruct"  # ← change to your model

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": 'Reply with {"ok": true} only.'}],
        max_tokens=16,
        temperature=0.0,
    )
    print("✓ Model OK:", resp.choices[0].message.content)
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    if hasattr(e, "status_code"):
        print(f"  HTTP {e.status_code}")
    if hasattr(e, "body"):
        print(f"  Body: {e.body}")

# ── 4. Confirm credit balance ──────────────────────────────────────────────────
# (OpenRouter silently fails on $0 balance for some models)
import urllib.request

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/auth/key",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
)
with urllib.request.urlopen(req) as r:
    info = json.loads(r.read())
    print(
        f"✓ Account: limit={info['data'].get('limit')}  "
        f"usage={info['data'].get('usage')}  "
        f"free_tier={info['data'].get('is_free_tier')}"
    )
