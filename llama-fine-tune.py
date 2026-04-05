#!/usr/bin/env python3
"""
LoRA Fine-tuning Script with Robust Model Path Resolution + Resume + Evaluation
"""

import os
import datetime
import warnings
import torch

# Suppress noisy library version mismatch warnings that don't affect functionality
warnings.filterwarnings("ignore", category=Warning, module="requests")
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_utils import get_last_checkpoint

def ts(stage: str):
    """Print a timestamped stage label."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {stage}")

class ProgressTimestampCallback(TrainerCallback):
    """Prints a timestamp every time training crosses a 5% progress threshold."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.last_pct_logged = -1

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        pct = int((state.global_step / self.total_steps) * 100)
        bucket = (pct // 5) * 5          # round down to nearest 5
        if bucket > self.last_pct_logged:
            self.last_pct_logged = bucket
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Training progress: {bucket}%  (step {state.global_step}/{self.total_steps})")

# =========================
# MODEL PATH RESOLUTION
# =========================
def resolve_model_path():
    paths = [
        os.path.expanduser("~/models/llama2"),
        "/Users/ansarmuhammad/Library/CloudStorage/OneDrive-10Pearls/llama/new models-copy from cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
        "/Users/ansarmuhammad/Library/CloudStorage/OneDrive-10Pearls/llama/new models-copy from cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf",  # fallback to parent dir
        "meta-llama/Llama-2-7b-chat-hf"  # final fallback
    ]

    for path in paths:
        if os.path.exists(path):
            print(f"Using model from: {path}")
            return path

    raise ValueError("No valid model path found.")

ts("Script started")
MODEL_PATH = resolve_model_path()
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora-output")

BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
MAX_STEPS = 50
SAVE_STEPS = 5

# =========================
# LOAD MODEL
# =========================
ts("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
ts("Tokenizer loaded")

ts("Loading model weights (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
ts("Model weights loaded")

# =========================
# APPLY LORA
# =========================
ts("Applying LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
ts("LoRA applied")

# =========================
# LOAD DATA
# =========================
ts("Loading dataset...")

dataset = load_dataset("mrdbourke/FoodExtract-1k")
dataset = dataset['train'].select(range(200))

# Inspect actual column names
print("Dataset columns:", dataset.column_names)
print("Sample row:", dataset[0])

# Detect text and label columns dynamically
_cols = dataset.column_names
TEXT_COL = next(
    (c for c in _cols if c in ['text', 'sequence', 'sentence', 'input', 'context', 'tokens', 'caption']),
    _cols[0]
)
LABEL_COL = next(
    (c for c in _cols if c in [
        'label', 'labels', 'output', 'answer', 'entities', 'ner_tags', 'tags',
        'gpt-oss-120b-label-condensed', 'gpt-oss-120b-label', 'class_label'
    ]),
    _cols[1] if len(_cols) > 1 else _cols[0]
)
print(f"Using text column: '{TEXT_COL}', label column: '{LABEL_COL}'")
ts("Dataset loaded and columns detected")

# FORMAT
def format_example(example):
    text_val = example[TEXT_COL]
    label_val = example[LABEL_COL]
    # Handle list-type columns (e.g. token lists)
    if isinstance(text_val, list):
        text_val = ' '.join(str(t) for t in text_val)
    if isinstance(label_val, list):
        label_val = ' '.join(str(l) for l in label_val)
    return {
        "text": f"Input: {text_val}\nOutput: {label_val}"
    }

ts("Formatting examples...")
dataset = dataset.map(format_example)
ts("Examples formatted")

# =========================
# TOKENIZATION
# =========================
def tokenize(example):
    return tokenizer(
        example['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )


ts("Tokenizing dataset...")
tokenized = dataset.map(tokenize, batched=True)
ts("Tokenization complete")

# =========================
# COLLATOR
# =========================
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =========================
# TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    fp16=False,
    report_to="none",
    dataloader_pin_memory=False,
)

# =========================
# RESUME LOGIC
# =========================
last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

if last_checkpoint:
    ts(f"Resuming from checkpoint: {last_checkpoint}")
else:
    ts("Starting fresh training")

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=collator,
    callbacks=[ProgressTimestampCallback(total_steps=MAX_STEPS)],
)

# =========================
# TRAIN
# =========================
ts("Training started")
trainer.train(resume_from_checkpoint=last_checkpoint)
ts("Training complete")

# =========================
# QUICK EVAL
# =========================
ts("Starting evaluation")
print("\n=== QUICK EVALUATION ===")

model.eval()
samples = dataset.select(range(3))

for i, sample in enumerate(samples):
    prompt = f"Input: {sample['text'].split('Output:')[0].replace('Input: ', '').strip()}\nOutput:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n--- SAMPLE {i+1} ---")
    print("INPUT:")
    print(prompt)
    print("\nPREDICTION:")
    print(prediction)
    print("\nTARGET:")
    print(sample['text'])

ts("Evaluation complete")

# =========================
# SAVE
# =========================
ts("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

ts("Done — all finished")
