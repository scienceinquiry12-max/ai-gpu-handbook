"""Fine-tune an open LLM with QLoRA on a single consumer GPU.

pip install -r requirements.txt
python train.py

Defaults use a tiny model so the script runs on ~8 GB of VRAM. Change MODEL_ID to
"Qwen/Qwen2.5-7B-Instruct" or "meta-llama/Llama-3.1-8B-Instruct" on a 16–24 GB GPU.

Guide: https://tensorcore.si/what-is-fine-tuning/
"""
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "trl-lib/Capybara"  # chat-format examples ("messages" column)

# Load the base model in 4-bit NF4: this is the "Q" in QLoRA.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Train small low-rank adapters on every linear layer; the 4-bit base stays frozen.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

dataset = load_dataset(DATASET, split="train[:2000]")

args = SFTConfig(
    output_dir="qlora-out",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    max_length=1024,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model("qlora-out/adapter")  # saves only the LoRA adapter (a few MB)
print("Saved adapter to qlora-out/adapter")
