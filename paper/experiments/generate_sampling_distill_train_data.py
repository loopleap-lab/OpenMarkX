import argparse
import os
import random
import json
from typing import Dict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    set_seed,
)

from watermarks.aar.aar_watermark import AarWatermark
from watermarks.kgw.watermark_processor import WatermarkLogitsProcessor
from watermarks.kth.kth_watermark import KTHWatermark
from watermarks.watermark_types import WatermarkType


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--tokenizer_name", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--dataset_split", type=str, default="train")
parser.add_argument("--dataset_num_skip", type=int, default=0)
parser.add_argument("--data_field", type=str, default="text")
parser.add_argument("--num_samples", type=int, required=True)
parser.add_argument("--min_new_tokens", type=int, default=256)
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--prompt_length", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--streaming", action="store_true", default=True)
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--output_train_file", type=str, required=True)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--watermark_config_file", type=str, required=True)
parser.add_argument("--save_interval", type=int, default=64000)
parser.add_argument("--dataloader_batch_size", type=int, default=1000)

args = parser.parse_args()


def build_tokenizer_and_dataset(args):
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens(
                {
                    "pad_token": DEFAULT_PAD_TOKEN,
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.dataset_split,
        streaming=args.streaming,
    )

    if args.dataset_num_skip > 0:
        dataset = dataset.skip(args.dataset_num_skip)

    return tokenizer, dataset


def iter_dataset_batches(dataset, tokenizer, args):
    buffer_texts = []
    for example in dataset:
        if args.data_field not in example:
            continue
        buffer_texts.append(example[args.data_field])
        if len(buffer_texts) >= args.dataloader_batch_size:
            prompt = tokenizer(
                buffer_texts,
                truncation=True,
                padding=True,
                max_length=args.prompt_length,
                return_tensors="pt",
            )
            yield {
                "input_ids": prompt["input_ids"],
                "attention_mask": prompt["attention_mask"],
            }
            buffer_texts = []
    if buffer_texts:
        prompt = tokenizer(
            buffer_texts,
            truncation=True,
            padding=True,
            max_length=args.prompt_length,
            return_tensors="pt",
        )
        yield {
            "input_ids": prompt["input_ids"],
            "attention_mask": prompt["attention_mask"],
        }


def build_model_and_watermark(args, tokenizer):
    model_name = args.model_name
    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    if args.fp16:
        model = model.half()
    model.eval()

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens(
                {
                    "pad_token": DEFAULT_PAD_TOKEN,
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )

    with open(args.watermark_config_file, "r") as f:
        watermark_config = json.load(f)

    if isinstance(watermark_config, list):
        if len(watermark_config) == 0:
            raise ValueError(
                f"Watermark config file {args.watermark_config_file} contains an empty list."
            )
        watermark_config = watermark_config[0]

    wtype = watermark_config["type"]

    def is_type(tname):
        if isinstance(wtype, WatermarkType):
            return wtype == getattr(WatermarkType, tname)
        if isinstance(wtype, str):
            return wtype.upper() == tname
        return False

    if is_type("AAR"):
        watermark = AarWatermark(
            vocab_size=len(tokenizer),
            k=watermark_config["k"],
            seed=watermark_config["seed"],
            device=device,
        )
        do_sample = False
        type_name = "AAR"
    elif is_type("KGW"):
        watermark = WatermarkLogitsProcessor(
            vocab=tokenizer.get_vocab().values(),
            gamma=watermark_config["gamma"],
            delta=watermark_config["delta"],
            seeding_scheme=watermark_config["seeding_scheme"],
            device=device,
        )
        do_sample = True
        type_name = "KGW"
    elif is_type("KTH"):
        watermark = KTHWatermark(
            vocab_size=len(tokenizer),
            key_len=watermark_config["key_len"],
            seed=watermark_config["seed"],
            num_shifts=watermark_config["num_shifts"],
            device=device,
        )
        do_sample = False
        type_name = "KTH"
    else:
        raise ValueError(f"Invalid watermark type {wtype}")

    watermark_config_vars = {}
    try:
        for k, v in vars(watermark).items():
            if isinstance(v, (str, int, float, bool, list)):
                watermark_config_vars[k] = v
    except Exception as e:
        print(f"Error loading watermark config vars for model {model_name}: {e}")

    simplified_model_name = [s for s in model_name.split("/") if s][-1]
    simplified_model_name = type_name + "_" + simplified_model_name

    return (
        model,
        watermark,
        do_sample,
        watermark_config,
        watermark_config_vars,
        simplified_model_name,
    )


def main(args):
    if os.path.exists(args.output_file) and not args.overwrite_output_file:
        raise ValueError(
            f"Output file {args.output_file} already exists and overwrite_output_file is False"
        )

    if os.path.exists(args.output_train_file) and not args.overwrite_output_file:
        raise ValueError(
            f"Output file {args.output_train_file} already exists and overwrite_output_file is False"
        )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_train_file), exist_ok=True)

    tokenizer, dataset = build_tokenizer_and_dataset(args)

    (
        model,
        watermark,
        do_sample,
        watermark_config,
        watermark_config_vars,
        simplified_model_name,
    ) = build_model_and_watermark(args, tokenizer)

    print(f"Generating samples for model {simplified_model_name}")

    metadata = {
        "model_name": simplified_model_name,
        "watermark_type": watermark_config["type"]
        if isinstance(watermark_config["type"], str)
        else str(watermark_config["type"]),
        "watermark_config": watermark_config,
        "watermark_config_vars": watermark_config_vars,
    }
    metadata.update(vars(args))

    total_generated = 0
    prev_save = 0

    with open(args.output_train_file, "w", encoding="utf-8") as train_f:
        for batch in tqdm(
            iter_dataset_batches(dataset, tokenizer, args), desc="Sampling"
        ):
            if total_generated >= args.num_samples:
                break

            input_ids_batch = batch["input_ids"]
            attention_mask_batch = batch["attention_mask"]

            batch_size_loader = input_ids_batch.shape[0]
            for start_idx in range(0, batch_size_loader, args.batch_size):
                if total_generated >= args.num_samples:
                    break

                end_idx = min(start_idx + args.batch_size, batch_size_loader)
                input_ids = input_ids_batch[start_idx:end_idx].to(device)
                attention_mask = attention_mask_batch[start_idx:end_idx].to(device)

                if isinstance(watermark, KTHWatermark) and getattr(
                    watermark, "num_shifts", 1
                ) > 1:
                    watermark.cur_shift = random.choice(watermark.possible_shifts)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        min_new_tokens=args.min_new_tokens,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        logits_processor=LogitsProcessorList([watermark]),
                        pad_token_id=tokenizer.eos_token_id,
                    )

                n_input_tokens = input_ids.shape[1]
                generated = tokenizer.batch_decode(
                    outputs[:, n_input_tokens:], skip_special_tokens=True
                )

                for text in generated:
                    if total_generated >= args.num_samples:
                        break
                    line = json.dumps({"text": text}, ensure_ascii=False)
                    train_f.write(line + "\n")
                    total_generated += 1

                    if total_generated >= prev_save + args.save_interval:
                        prev_save = total_generated
                        save_filename = (
                            f"{args.output_file.rsplit('.', maxsplit=1)[0]}"
                            f"_save-{prev_save}.json"
                        )
                        snapshot = dict(metadata)
                        snapshot["num_generated"] = total_generated
                        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
                        with open(save_filename, "w", encoding="utf-8") as sf:
                            print(
                                f"Writing metadata snapshot to {save_filename}, "
                                f"num_generated={total_generated}"
                            )
                            json.dump(snapshot, sf, indent=4, ensure_ascii=False)

    del model
    torch.cuda.empty_cache()
    del watermark

    final_output = dict(metadata)
    final_output["num_generated"] = total_generated
    with open(args.output_file, "w", encoding="utf-8") as f:
        print(
            f"Finished. Generated {total_generated} samples. "
            f"Writing metadata to {args.output_file}"
        )
        json.dump(final_output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main(args)
