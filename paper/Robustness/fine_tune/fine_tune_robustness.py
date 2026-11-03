import os
import argparse
import logging
from transformers import (
    logging as hf_logging,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import torch

hf_logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_steps", type=int, default=2500)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--data_percent", type=float, default=100.0)
    return parser.parse_args()


def format_math(batch):
    questions = batch["question"]
    answers = batch["expected_answer"]
    q_tag = "[MathQuestion]"
    a_tag = "[MathAnswer]"
    texts = [
        f"{q_tag}\n{q.strip()}\n\n{a_tag}\n{a.strip()}\n"
        for q, a in zip(questions, answers)
    ]
    return {"text": texts}


def format_code(batch):
    inputs = batch["input"]
    outputs = batch["output"]
    q_tag = "[Question]"
    a_tag = "[Answer]"
    texts = [
        f"{q_tag}\n{inp.strip()}\n\n{a_tag}\n{out.strip()}\n"
        for inp, out in zip(inputs, outputs)
    ]
    return {"text": texts}


def format_openwebtext(batch, eos_token):
    texts_raw = batch["text"]
    texts = [t.strip() + eos_token for t in texts_raw]
    return {"text": texts}


def tokenize_and_group(raw_ds, tokenizer, block_size):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False)

    tokenized_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=8,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    def group_fn(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_ds = tokenized_ds.map(
        group_fn,
        batched=True,
        num_proc=8,
        desc=f"Grouping texts into blocks of {block_size}",
    )
    lm_ds.set_format(type="torch", columns=["input_ids"])
    return lm_ds


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.dataset_name in ["openwebtext", "Skylion007/openwebtext"]:
        pct = max(0.0, min(args.data_percent, 100.0))
        pct_int = int(pct)
        if pct_int >= 100:
            split = "train"
        else:
            split = f"train[:{pct_int}%]"
        logger.info(f"Loading {args.dataset_name} with split={split}")
        raw_ds = load_dataset(
            args.dataset_name,
            split=split,
        )
    else:
        raw_ds = load_dataset(
            args.dataset_name,
            split="train",
        )

    raw_ds = raw_ds.shuffle(seed=42)

    if args.max_train_samples > 0:
        raw_ds = raw_ds.select(range(min(len(raw_ds), args.max_train_samples)))
    elif args.data_percent < 100.0 and args.dataset_name not in [
        "openwebtext",
        "Skylion007/openwebtext",
    ]:
        pct = max(0.0, min(args.data_percent, 100.0))
        n = int(len(raw_ds) * pct / 100.0)
        raw_ds = raw_ds.select(range(n))

    if args.dataset_name == "nvidia/OpenMathInstruct-1":
        raw_ds = raw_ds.map(
            format_math,
            batched=True,
            num_proc=8,
            batch_size=1000,
            remove_columns=raw_ds.column_names,
        )
    elif args.dataset_name == "nvidia/OpenCodeInstruct":
        if not {"input", "output"}.issubset(raw_ds.column_names):
            raise ValueError("Dataset must contain input and output")
        raw_ds = raw_ds.map(
            format_code,
            batched=True,
            num_proc=8,
            batch_size=1000,
            remove_columns=raw_ds.column_names,
        )
    elif args.dataset_name in ["openwebtext", "Skylion007/openwebtext"]:
        if "text" not in raw_ds.column_names:
            raise ValueError("openwebtext must contain text column")
        raw_ds = raw_ds.map(
            lambda batch: format_openwebtext(batch, tokenizer.eos_token or ""),
            batched=True,
            num_proc=8,
            batch_size=1000,
            remove_columns=raw_ds.column_names,
        )
    else:
        if "text" not in raw_ds.column_names:
            raise ValueError("Dataset must contain text column")

    lm_ds = tokenize_and_group(raw_ds, tokenizer, args.block_size)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    if dtype is None:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)

    sample = lm_ds[0]["input_ids"]
    logger.info("Example:\n" + tokenizer.decode(sample[:200]))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to=args.report_to,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)

    if last_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("No checkpoint found, starting fresh training")
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
