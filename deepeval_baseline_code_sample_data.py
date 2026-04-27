# %% [markdown] {"execution":{"iopub.status.busy":"2026-04-26T10:02:33.765987Z","iopub.execute_input":"2026-04-26T10:02:33.766223Z","iopub.status.idle":"2026-04-26T10:03:08.505984Z","shell.execute_reply.started":"2026-04-26T10:02:33.766201Z","shell.execute_reply":"2026-04-26T10:03:08.504904Z"}}
# # ============================================================
# # CELL 1 — Environment + GPU check
# # ============================================================
#  
# import torch
# import sys
#  
# print("=" * 60)
# print("KAGGLE ENVIRONMENT CHECK")
# print("=" * 60)
# print(f"\nPython:  {sys.version.split()[0]}")
# print(f"PyTorch: {torch.__version__}")
# print(f"CUDA:    {torch.cuda.is_available()}")
#  
# if torch.cuda.is_available():
#     n_gpu = torch.cuda.device_count()
#     print(f"\nGPU count: {n_gpu}")
#     for i in range(n_gpu):
#         name = torch.cuda.get_device_name(i)
#         vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
#         print(f"  GPU {i}: {name} ({vram:.1f} GB)")
#  
#     gpu_cap = torch.cuda.get_device_capability(0)
#     print(f"\nGPU compute capability: sm_{gpu_cap[0]}{gpu_cap[1]}")
# else:
#     print("\n✗ NO GPU — enable from sidebar")
#  
# # Verify Kaggle secrets
# try:
#     from kaggle_secrets import UserSecretsClient
#     us = UserSecretsClient()
#     _ = us.get_secret("HF_TOKEN")
#     _ = us.get_secret("GROQ_API_KEY")
#     print("\n✓ HF_TOKEN + GROQ_API_KEY available in Kaggle secrets")
# except Exception as e:
#     print(f"\n✗ Secrets issue: {e}")
#     print("  → Add-ons → Secrets → enable HF_TOKEN and GROQ_API_KEY")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-26T10:03:08.507787Z","iopub.execute_input":"2026-04-26T10:03:08.508104Z","iopub.status.idle":"2026-04-26T10:04:12.325356Z","shell.execute_reply.started":"2026-04-26T10:03:08.508083Z","shell.execute_reply":"2026-04-26T10:04:12.324042Z"}}
# ============================================================
# CELL 2 — Install Unsloth stack (proven for Gemma-4 E2B)
# ============================================================
 
# Unsloth is REQUIRED for Gemma-4 E2B-it due to:
#   - Per-Layer Embeddings (PLE) architecture
#   - Shared KV cache across 20 layers
#   - Gemma4ClippableLinear custom layer class
# Direct transformers + bitsandbytes causes OOM on T4 (verified in v1 debug)
 
#!pip install -q -U unsloth
#!pip install -q -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
 
# Supporting libraries
#!pip install -q -U trl peft accelerate bitsandbytes datasets
#!pip install -q -U huggingface_hub sentencepiece protobuf
 
# Evaluation framework
#!pip install -q -U deepeval litellm
 
# Version report
import transformers, peft, trl, torch, deepeval

print("=" * 60)
print("VERSION CHECK")
print("=" * 60)
print(f"  transformers:  {transformers.__version__}")
print(f"  peft:          {peft.__version__}")
print(f"  trl:           {trl.__version__}")
print(f"  deepeval:      {deepeval.__version__}")
print(f"  torch:         {torch.__version__}")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-26T10:04:12.325936Z","iopub.status.idle":"2026-04-26T10:04:12.326187Z","shell.execute_reply.started":"2026-04-26T10:04:12.326075Z","shell.execute_reply":"2026-04-26T10:04:12.326090Z"}}
# ============================================================
# CELL 3 — Authentication (HuggingFace only; judge is local Ollama)
# ============================================================

import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")

os.environ['LITELLM_SUPPRESS_DEBUG_INFO'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✓ HuggingFace authenticated (needed for model weights download)")
else:
    print("ℹ HF_TOKEN not set — model weight download may require authentication")

print("✓ Using local Ollama as judge — no API key required")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-26T10:04:12.326904Z","iopub.status.idle":"2026-04-26T10:04:12.327171Z","shell.execute_reply.started":"2026-04-26T10:04:12.327051Z","shell.execute_reply":"2026-04-26T10:04:12.327066Z"}}
# ============================================================
# CELL 4 — Config (local sample data)
# ============================================================

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Model — use Unsloth's pre-quantized variant (faster load)
    model_id: str = "unsloth/gemma-4-E2B-it"

    # Local sample data files (relative to this script's directory)
    data_dir: str = str(Path(__file__).parent / "data")
    train_file: str = "train.json"
    test_file: str = "test.json"

    # Eval settings — capped at test set size
    eval_n_samples: int = 10

    # Generation settings (deterministic for reproducibility)
    max_new_tokens: int = 256
    max_input_length: int = 1024

    # Judge — local Ollama (change model name to match what you have: ollama list)
    eval_model: str = "ollama/gemma:2b"
    ollama_base_url: str = "http://localhost:11434"
    eval_threshold: float = 0.7

    # Output directory (local)
    base_dir: str = str(Path(__file__).parent / "car_repair_slm")

CONFIG = Config()
BASE = Path(CONFIG.base_dir)
RESULTS = BASE / "results"
LOGS = BASE / "logs"
for d in [BASE, RESULTS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

BASELINE_ANSWERS = RESULTS / "baseline_answers_v2.json"
BASELINE_SCORES = RESULTS / "baseline_scores_v2.json"

print(f"Model:    {CONFIG.model_id}")
print(f"Data dir: {CONFIG.data_dir}")
print(f"Judge:    {CONFIG.eval_model}")
print(f"Outputs:  {BASE}")
print(f"\n✓ Config locked")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-26T10:04:12.328258Z","iopub.status.idle":"2026-04-26T10:04:12.328524Z","shell.execute_reply.started":"2026-04-26T10:04:12.328386Z","shell.execute_reply":"2026-04-26T10:04:12.328404Z"}}
# ============================================================
# CELL 5 — Load train.json + test.json from local sample files
# ============================================================

import json
from pathlib import Path

data_dir = Path(CONFIG.data_dir)

# train.json (reserved for fine-tuning, not used in this baseline)
train_path = data_dir / CONFIG.train_file
print(f"Loading {train_path}...")
with open(train_path) as f:
    train_data = json.load(f)

# test.json (used for baseline eval)
test_path = data_dir / CONFIG.test_file
print(f"Loading {test_path}...")
with open(test_path) as f:
    test_data = json.load(f)

# Cap eval size at configured limit
eval_data = test_data[:CONFIG.eval_n_samples] if len(test_data) > CONFIG.eval_n_samples else test_data

print(f"\n✓ Train pool:  {len(train_data)} samples (reserved for fine-tuning)")
print(f"✓ Test total:  {len(test_data)} samples")
print(f"✓ Eval set:    {len(eval_data)} samples")
print(f"\nFirst eval item keys: {list(eval_data[0].keys())}")
print(f"\nPreview:")
print(json.dumps(eval_data[0], indent=2, ensure_ascii=False)[:500])

# %% [code] {"execution":{"iopub.status.busy":"2026-04-26T10:04:12.329156Z","iopub.status.idle":"2026-04-26T10:04:12.329371Z","shell.execute_reply.started":"2026-04-26T10:04:12.329276Z","shell.execute_reply":"2026-04-26T10:04:12.329289Z"}}
# ============================================================
# CELL 6 — Schema detection (auto-detects field names)
# ============================================================
 
sample = eval_data[0]
 
Q_CANDIDATES = ['question', 'input', 'query', 'prompt', 'instruction']
A_CANDIDATES = ['answer', 'expected_answer', 'output', 'response', 'completion']
C_CANDIDATES = ['context', 'retrieval_context', 'reference', 'passage', 'background']
 
def find_key(item, candidates):
    for k in candidates:
        if k in item and item[k]:
            return k
    return None
 
Q_KEY = find_key(sample, Q_CANDIDATES)
A_KEY = find_key(sample, A_CANDIDATES)
C_KEY = find_key(sample, C_CANDIDATES)
 
assert Q_KEY and A_KEY, f"Missing question/answer field. Got keys: {list(sample.keys())}"
 
has_context = bool(C_KEY) and sum(1 for it in eval_data if it.get(C_KEY)) >= len(eval_data) * 0.8
 
print(f"Question field: '{Q_KEY}'")
print(f"Answer field:   '{A_KEY}'")
print(f"Context field:  '{C_KEY}'  usable={has_context}")
 
metric_list = ["AnswerRelevancy", "GEval-Correctness"]
if has_context:
    metric_list += ["Faithfulness", "ContextualRelevancy"]
print(f"\nMetrics selected: {metric_list}")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-26T10:04:12.330254Z","iopub.status.idle":"2026-04-26T10:04:12.330478Z","shell.execute_reply.started":"2026-04-26T10:04:12.330383Z","shell.execute_reply":"2026-04-26T10:04:12.330395Z"}}
# ============================================================
# CELL 7 — Judge setup (LiteLLMJudge — UNCHANGED from v1, proven)
# ============================================================
 
import os, time
import litellm
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
 
litellm.suppress_debug_info = True
 
 
class LiteLLMJudge(DeepEvalBaseLLM):
    """
    Local Ollama judge via LiteLLM.
    - No API key required
    - No rate limiting (local inference)
    - JSON retry with expanded max_tokens
    - Handles connection errors with backoff
    """

    def __init__(self, model_name, api_base="http://localhost:11434", max_tokens=1024):
        self.model_name = model_name
        self.api_base = api_base
        self.max_tokens = max_tokens

    def _call(self, prompt, schema=None, retries=5):
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "api_base": self.api_base,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if schema is not None:
            kwargs["format"] = "json"

        last_err = None
        for attempt in range(retries):
            try:
                resp = litellm.completion(**kwargs)
                text = resp.choices[0].message.content.strip()

                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                        if text.startswith("json"):
                            text = text[4:]
                        text = text.strip()

                if schema is not None:
                    try:
                        return schema.model_validate_json(text)
                    except Exception as pe:
                        last_err = pe
                        if attempt < retries - 1:
                            kwargs["max_tokens"] = min(kwargs["max_tokens"] * 2, 4096)
                            continue
                        raise
                return text

            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "503" in msg or "unavailable" in msg or "connection" in msg:
                    wait = min(5 * (2 ** attempt), 30)
                    print(f"\n  [attempt {attempt+1}] Ollama unavailable, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                if "validation" in msg or "json" in msg:
                    if attempt < retries - 1:
                        kwargs["max_tokens"] = min(kwargs["max_tokens"] * 2, 4096)
                        continue
                raise
        raise RuntimeError(f"Judge failed after {retries} retries: {last_err}")

    def load_model(self):
        return self.model_name

    def generate(self, prompt, schema=None):
        return self._call(prompt, schema)

    async def a_generate(self, prompt, schema=None):
        return self._call(prompt, schema)

    def generate_raw_response(self, prompt, top_logprobs=None, schema=None):
        text = self._call(prompt, schema)
        class _FakeChoice:
            def __init__(self, t): self.message = type("M", (), {"content": t})
        class _FakeResp:
            def __init__(self, t):
                self.choices = [_FakeChoice(t)]
                self.usage = type("U", (), {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0})
        return _FakeResp(text if isinstance(text, str) else str(text)), 0.0

    async def a_generate_raw_response(self, prompt, top_logprobs=None, schema=None):
        return self.generate_raw_response(prompt, top_logprobs, schema)

    def get_model_name(self):
        return self.model_name


# Build judge
judge = LiteLLMJudge(
    model_name=CONFIG.eval_model,
    api_base=CONFIG.ollama_base_url,
    max_tokens=1024,
)

# Smoke test
print(f"Smoke-testing Ollama judge ({CONFIG.eval_model})...")
probe = judge.generate("Reply with exactly one word: OK")
print(f"  Judge reply: {str(probe)[:80]}")
print("  ✓ Judge working\n")
 
# Build metrics
metrics = [
    AnswerRelevancyMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
    GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the actual_output is a factually correct, technically accurate "
            "answer to the input question, using expected_output as ground truth. For car repair, "
            "check: correct parts/components, correct diagnostic reasoning, correct procedures, "
            "safe advice. Heavily penalize vague, generic, or incorrect technical claims."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=CONFIG.eval_threshold,
        model=judge,
        async_mode=False,
    ),
]
 
if has_context:
    from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric
    metrics += [
        FaithfulnessMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
        ContextualRelevancyMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
    ]
 
print(f"Judge:   {CONFIG.eval_model} (local Ollama, no rate limits)")
print(f"Metrics: {len(metrics)} (sequential mode)")

# %% [code]
# ============================================================
# CELL 8 — Load model via standard transformers (no Unsloth)
# ============================================================

import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Pick the best available device: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Loading {CONFIG.model_id} via transformers (device: {device})...")
print("First-time download: ~3-5 min for ~3.5 GB weights\n")

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG.model_id,
    token=HF_TOKEN or None,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN or None,
)

model.eval()

print(f"  Device: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {alloc:.2f} / {total_mem:.1f} GB")

print("\n✓ Base model loaded (no adapter — this is pre-fine-tune baseline)")

# %% [code]
# ============================================================
# CELL 9 — Generate baseline answers (FIXED tokenizer call)
# ============================================================
 
import json
from tqdm.auto import tqdm
 
SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)
 
def build_prompt(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
 
@torch.no_grad()
def generate(question: str) -> str:
    prompt = build_prompt(question)
    # FIX: multimodal tokenizer requires text= keyword (not positional)
    inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CONFIG.max_input_length,
    ).to(model.device)
 
    out = model.generate(
        **inputs,
        max_new_tokens=CONFIG.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
 
# --- Smoke test ---
print("Smoke test on sample 0...\n")
q0 = eval_data[0][Q_KEY]
a0 = generate(q0)
print(f"Q:        {q0[:200]}")
print(f"Generated:{a0[:400]}")
print(f"Expected: {eval_data[0][A_KEY][:200]}")
print("\n" + "=" * 60)
 
# --- Full baseline generation ---
print(f"\nGenerating baselines for {len(eval_data)} questions...")
baseline_results = []
for item in tqdm(eval_data):
    q = item[Q_KEY]
    expected = item[A_KEY]
    ctx = item.get(C_KEY, "") if C_KEY else ""
    gen = generate(q)
    baseline_results.append({
        "question": q,
        "expected_answer": expected,
        "generated_answer": gen,
        "context": ctx,
    })
 
with open(BASELINE_ANSWERS, 'w') as f:
    json.dump(baseline_results, f, indent=2, ensure_ascii=False)
 
print(f"\nSaved {len(baseline_results)} baseline answers → {BASELINE_ANSWERS}")

# %% [code]
# ============================================================
# CELL 10 — Score with DeepEval (FIXED: to_str helper for list/str)
# ============================================================

from collections import defaultdict
from tqdm.auto import tqdm
import json

def to_str(value):
    """Convert answer to string — handle list, str, None."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if item)
    return str(value)

# Build test cases (with format normalization)
test_cases = []
for r in baseline_results:
    ctx_raw = r.get("context", "")
    ctx_str = to_str(ctx_raw)
    ctx_list = [ctx_str] if (has_context and ctx_str) else None
    
    test_cases.append(LLMTestCase(
        input=to_str(r["question"]),
        actual_output=to_str(r["generated_answer"]),
        expected_output=to_str(r["expected_answer"]),
        retrieval_context=ctx_list,
    ))

total_calls = len(test_cases) * len(metrics)
est_min = (total_calls * 6.0) / 60
print(f"Sequential eval: {len(test_cases)} cases × {len(metrics)} metrics = {total_calls} judge calls")
print(f"Rate-limited (~6s/call) → ~{est_min:.0f} min\n")

# Checkpoint setup
BASELINE_CHECKPOINT = RESULTS / "baseline_checkpoint_v2.jsonl"
scores_by_metric = defaultdict(list)
per_case = []
failed_count = 0

# Resume support
completed_indices = set()
if BASELINE_CHECKPOINT.exists():
    with open(BASELINE_CHECKPOINT) as f:
        for line in f:
            try:
                entry = json.loads(line)
                completed_indices.add(entry["idx"])
                per_case.append(entry)
                for k, v in entry.get("scores", {}).items():
                    if isinstance(v, (int, float)):
                        scores_by_metric[k].append(v)
            except Exception:
                pass
    if completed_indices:
        print(f"Resuming from checkpoint: {len(completed_indices)} cases already scored\n")

# Score each case
with open(BASELINE_CHECKPOINT, 'a') as ckpt_file:
    for i, tc in enumerate(tqdm(test_cases, desc="Scoring")):
        if i in completed_indices:
            continue

        case_scores = {}
        case_failed = False

        for metric in metrics:
            mname = getattr(metric, 'name', None) or metric.__class__.__name__
            try:
                metric.measure(tc)
                score = getattr(metric, 'score', None)
                if score is not None:
                    scores_by_metric[mname].append(score)
                    case_scores[mname] = round(score, 3)
            except Exception as e:
                print(f"\n  [case {i}] {mname} failed: {str(e)[:150]}")
                case_failed = True

        if case_failed and not case_scores:
            failed_count += 1

        entry = {
            "idx": i,
            "question": to_str(baseline_results[i]["question"])[:150],
            "scores": case_scores,
        }
        per_case.append(entry)

        ckpt_file.write(json.dumps(entry) + "\n")
        ckpt_file.flush()

# Aggregate
summary = {
    "model": CONFIG.model_id,
    "dataset": str(Path(CONFIG.data_dir) / CONFIG.test_file),
    "judge": CONFIG.eval_model,
    "n_samples": len(test_cases),
    "n_failed": failed_count,
    "threshold": CONFIG.eval_threshold,
    "metrics": {},
    "per_case": sorted(per_case, key=lambda x: x["idx"]),
}

print("\n" + "=" * 70)
print(f"BASELINE SCORES — {CONFIG.model_id} (pre-fine-tune, v2 dataset)")
print("=" * 70)
for name, vals in scores_by_metric.items():
    if not vals:
        continue
    avg = sum(vals) / len(vals)
    pass_rate = sum(1 for v in vals if v >= CONFIG.eval_threshold) / len(vals)
    summary["metrics"][name] = {
        "avg_score": round(avg, 4),
        "pass_rate": round(pass_rate, 4),
        "n": len(vals),
    }
    print(f"  {name:35s}  avg={avg:.3f}  pass_rate={pass_rate:.1%}  (n={len(vals)})")

if failed_count:
    print(f"\n  ⚠ {failed_count} cases failed judging")
print("=" * 70)

with open(BASELINE_SCORES, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Saved summary → {BASELINE_SCORES}")

# %% [code]
# ============================================================
# CELL 11 — Zip artifacts for download (optional but recommended)
# ============================================================
 
import shutil
from pathlib import Path
 
zip_base = "/kaggle/working/baseline_v2_artifacts"
shutil.make_archive(
    base_name=zip_base,
    format='zip',
    root_dir=str(RESULTS),
)
 
zip_file = Path(f"{zip_base}.zip")
size_mb = zip_file.stat().st_size / 1024 / 1024
 
print(f"✓ Zip created: {zip_file}")
print(f"  Size: {size_mb:.2f} MB")
print(f"\nNEXT STEPS:")
print(f"  1. Kaggle right sidebar → Output section")
print(f"  2. Download baseline_v2_artifacts.zip")
print(f"  3. Backup to Drive + laptop")
print(f"\nThese baseline scores are the reference for post-fine-tune comparison.")