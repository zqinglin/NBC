import argparse
import os
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, Trainer, DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model


DATASET_MAPPING = {
    "logic": {
        "path": "microsoft/orca-math-word-problems-200k",
        "subset": None,
        "prompt_col": "question",
        "response_col": "answer",
    },
    "coder": {
        "path": "nickrosh/Evol-Instruct-Code-80k-v1",
        "subset": None,
        "prompt_col": "instruction",
        "response_col": "output",
    },
    "persona": {
        "path": "proj-persona/PersonaHub",
        "subset": "instruction",
        "prompt_col": "input persona",
        "response_col": "synthesized text",
    },
}


def format_examples(ds, prompt_col, response_col):
    out = []
    for ex in ds:
        p = ex.get(prompt_col, "")
        r = ex.get(response_col, "")
        if isinstance(r, list):
            if len(r) and isinstance(r[0], dict):
                r = "\n".join([str(m.get("content", "")) for m in r])
            else:
                r = "\n".join([str(x) for x in r])
        text = f"Instruction: {str(p)}\nResponse: {str(r)}"
        out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot_name", required=True)
    ap.add_argument("--dataset_name", default=None)
    ap.add_argument("--output_dir", default="./saved_slots")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--quant", default="16bit")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--num_train_epochs", type=int, default=1)
    args = ap.parse_args()
    slot = args.slot_name
    mapping = DATASET_MAPPING.get(slot)
    if mapping:
        if mapping.get("subset"):
            ds = load_dataset(mapping["path"], mapping["subset"], split="train")
        else:
            ds = load_dataset(mapping["path"], split="train")
        prompt_col = mapping["prompt_col"]
        response_col = mapping["response_col"]
    else:
        if args.dataset_name is None:
            raise RuntimeError("dataset_name is required for unknown slot")
        ds = load_dataset(args.dataset_name, split="train")
        cols = list(ds.features.keys())
        prompt_col = "instruction" if "instruction" in cols else cols[0]
        response_col = "output" if "output" in cols else cols[-1]
    texts = format_examples(ds, prompt_col, response_col)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if str(args.quant).lower() == "4bit":
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_cfg, device_map="auto")
    else:
        dt = torch.bfloat16 if str(args.dtype).lower() == "bf16" else torch.float16
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
        base.to(device)
    base.train(False)
    for p in base.parameters():
        p.requires_grad = False
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg)
    if slot not in getattr(model, "peft_config", {}):
        model.add_adapter(slot, lcfg)
    model.set_adapter(slot)
    ta_kwargs = {
        "output_dir": os.path.join(args.output_dir, "_tmp"),
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-4,
        "logging_steps": 10,
        "gradient_accumulation_steps": 1,
        "save_strategy": "no",
        "bf16": (str(args.dtype).lower() == "bf16"),
        "fp16": (str(args.dtype).lower() == "fp16"),
        "dataloader_num_workers": 2,
        "report_to": [],
    }
    if args.num_train_epochs and args.num_train_epochs > 0:
        ta_kwargs["num_train_epochs"] = args.num_train_epochs
    else:
        ta_kwargs["max_steps"] = args.max_steps
    train_args = TrainingArguments(**ta_kwargs)
    dataset = Dataset.from_dict({"text": texts})
    trainer = None
    try:
        trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=dataset, args=train_args, dataset_text_field="text", max_seq_length=1024)
    except TypeError:
        try:
            trainer = SFTTrainer(model=model, train_dataset=dataset, args=train_args, dataset_text_field="text", max_seq_length=1024)
        except TypeError:
            tokenized = dataset.map(lambda x: tok(x["text"], truncation=True, max_length=1024), batched=True, remove_columns=dataset.column_names)
            collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
            trainer = Trainer(model=model, args=train_args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    save_path = os.path.join(args.output_dir, slot)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)


if __name__ == "__main__":
    main()
