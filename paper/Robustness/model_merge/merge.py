import os
import subprocess
import shutil
import argparse
import torch
from transformers import AutoTokenizer
from colorama import init, Fore, Style
from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
from tqdm import tqdm

init(autoreset=True)

def clear_console():
    if os.name == "nt":
        subprocess.call("cls", shell=True)
    else:
        subprocess.call("clear", shell=True)

def slerp_torch(t, v0, v1, DOT_THRESHOLD=0.9995, eps=1e-10):
    if not (v0.is_floating_point() and v1.is_floating_point()):
        return v0
    orig_dtype = v0.dtype
    v0 = v0.to(torch.float32)
    v1 = v1.to(torch.float32)
    v0_norm = v0.norm()
    v1_norm = v1.norm()
    if v0_norm > eps:
        v0_unit = v0 / v0_norm
    else:
        v0_unit = v0
    if v1_norm > eps:
        v1_unit = v1 / v1_norm
    else:
        v1_unit = v1
    dot = (v0_unit * v1_unit).sum()
    if torch.abs(dot) > DOT_THRESHOLD:
        out = (1 - t) * v0 + t * v1
        return out.to(orig_dtype)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    out = s0 * v0 + s1 * v1
    return out.to(orig_dtype)

def list_model_shards(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} is not a directory")
    files = [f for f in os.listdir(path) if (f.endswith(".safetensors") or f.endswith(".bin"))]
    shards = [f for f in files if ("model" in f and "optimizer" not in f and "scheduler" not in f and "training_args" not in f and "fsdp" not in f)]
    if not shards:
        raise FileNotFoundError(f"No model shard files found in {path}")
    def shard_sort_key(filename):
        try:
            parts = filename.split("-")
            return int(parts[1])
        except Exception:
            return 0
    shards = sorted(shards, key=shard_sort_key)
    return shards

def load_shard(path):
    if path.endswith(".bin"):
        state = torch.load(path, map_location="cpu")
    elif path.endswith(".safetensors"):
        state = safe_load_file(path, device="cpu")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return state

def save_shard(state_dict, path):
    if path.endswith(".bin"):
        torch.save(state_dict, path)
    elif path.endswith(".safetensors"):
        safe_save_file(state_dict, path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def blend_models_sharded(model1_path, model2_path, output_path, alpha):
    print(f"{Fore.YELLOW}Listing shards...{Style.RESET_ALL}")
    shards1 = list_model_shards(model1_path)
    shards2 = list_model_shards(model2_path)
    print(f"{Fore.YELLOW}Loading full base model (model2) into memory...{Style.RESET_ALL}")
    model2_state = {}
    for shard_name in tqdm(shards2, desc="Loading model2 shards"):
        shard_path = os.path.join(model2_path, shard_name)
        state2 = load_shard(shard_path)
        model2_state.update(state2)
        del state2
    keys1_all = set()
    for shard_name in shards1:
        shard_path = os.path.join(model1_path, shard_name)
        state1_tmp = load_shard(shard_path)
        keys1_all.update(state1_tmp.keys())
        del state1_tmp
    missing_in_model2 = [k for k in keys1_all if k not in model2_state]
    if missing_in_model2:
        print(f"{Fore.LIGHTBLACK_EX}Warning: {len(missing_in_model2)} keys in model1 not found in model2. They will be copied from model1.{Style.RESET_ALL}")
    os.makedirs(output_path, exist_ok=True)
    print(f"{Fore.YELLOW}Blending shards...{Style.RESET_ALL}")
    t = 1 - alpha
    for shard_name in tqdm(shards1, desc="Blending model1 shards"):
        shard1_path = os.path.join(model1_path, shard_name)
        out_shard_path = os.path.join(output_path, shard_name)
        state1 = load_shard(shard1_path)
        blended_state = {}
        for k, v0 in state1.items():
            if k in model2_state:
                v1 = model2_state[k]
                blended_state[k] = slerp_torch(t, v0, v1)
            else:
                blended_state[k] = v0
        save_shard(blended_state, out_shard_path)
        del state1, blended_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "tokenizer.model",
        "added_tokens.json",
        "merges.txt",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    src_dir = model1_path if os.path.exists(os.path.join(model1_path, "config.json")) else model2_path
    for file_name in files_to_copy:
        src_file = os.path.join(src_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(output_path, file_name))
        else:
            print(f"{Fore.LIGHTBLACK_EX}Skipping {file_name}, not found.{Style.RESET_ALL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(src_dir)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"{Fore.LIGHTBLACK_EX}Tokenizer load/save failed: {e}{Style.RESET_ALL}")

def main():
    clear_console()
    parser = argparse.ArgumentParser(description="Blend two Huggingface LLMs using SLERP (sharded, low memory).")
    parser.add_argument("--model1", type=str, required=True, help="Path to the first model.")
    parser.add_argument("--model2", type=str, required=True, help="Path to the second model.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the blended model.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Interpolation factor (0.0-1.0). Default=0.5")
    args = parser.parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")
    blend_models_sharded(args.model1, args.model2, args.output, args.alpha)
    print(f"{Fore.CYAN}Blending complete! Model saved to {args.output}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
