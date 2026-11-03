import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import os

def quantize_int8_or_4bit(model_path, save_path, quant_type="int8", use_fp16=True):
    print(f"Quantizing to {quant_type.upper()} using bitsandbytes...")

    quant_config = None
    if quant_type == "int8":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=use_fp16
        )
    elif quant_type == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if use_fp16 else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # or "fp4"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Quantization complete.")

def quantize_gptq(model_path, save_path, bits=4, dataset="c4", group_size=128):
    from auto_gptq import AutoGPTQForCausalLM, GPTQConfig
    from transformers import AutoTokenizer

    print(f"Quantizing using GPTQ ({bits}-bit)...")
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=dataset,
        group_size=group_size,
        desc_act=False,
        disable_exllama=True
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=gptq_config,
        use_triton=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("GPTQ quantization complete.")

def quantize_awq(model_path, save_path, w_bit=4):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Quantizing using AWQ ({w_bit}-bit)...")
    model = AutoAWQForCausalLM.from_pretrained(model_path, fuse_layers=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.quantize(
        w_bit=w_bit,
        q_config="awq",
        fuse_layers=True,
        version="GEMM"
    )

    model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)
    print("AWQ quantization complete.")

def main():
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face LLM model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save quantized model")
    parser.add_argument("--quant_type", type=str, required=True,
                        choices=["int8", "4bit", "gptq", "awq"],
                        help="Quantization method")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 where applicable")
    parser.add_argument("--gptq_bits", type=int, default=4, help="GPTQ bit width (default: 4)")
    parser.add_argument("--gptq_dataset", type=str, default="c4", help="Dataset used to calibrate GPTQ")
    parser.add_argument("--gptq_group_size", type=int, default=128, help="GPTQ group size")
    parser.add_argument("--awq_bits", type=int, default=4, help="AWQ bit width")

    args = parser.parse_args()

    if args.quant_type in ["int8", "4bit"]:
        quantize_int8_or_4bit(
            model_path=args.model_path,
            save_path=args.save_path,
            quant_type=args.quant_type,
            use_fp16=not args.no_fp16
        )
    elif args.quant_type == "gptq":
        quantize_gptq(
            model_path=args.model_path,
            save_path=args.save_path,
            bits=args.gptq_bits,
            dataset=args.gptq_dataset,
            group_size=args.gptq_group_size
        )
    elif args.quant_type == "awq":
        quantize_awq(
            model_path=args.model_path,
            save_path=args.save_path,
            w_bit=args.awq_bits
        )

if __name__ == "__main__":
    main()
