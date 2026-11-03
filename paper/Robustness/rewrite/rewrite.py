import json
import time
import argparse
import nltk
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize.punkt import PunktSentenceTokenizer

# Download tokenizer data
nltk.download('punkt')
tokenizer = PunktSentenceTokenizer()


class DipperParaphraser:
    def __init__(self, model_name="kalpeshk2011/dipper-paraphraser-xxl"):
        start = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        self.model.eval()
        print(f"Model loaded in {time.time() - start:.2f}s")

    def paraphrase(self, text, lex_div=60, order_div=0, sent_interval=3, batch_size=4, **kwargs):
        lex_code = 100 - lex_div
        order_code = 100 - order_div

        sentences = tokenizer.tokenize(text)
        chunks = [
            " ".join(sentences[i:i + sent_interval])
            for i in range(0, len(sentences), sent_interval)
        ]
        prompts = [
            f"lexical = {lex_code}, order = {order_code} <sent> {chunk} </sent>"
            for chunk in chunks
        ]

        output_text = ""
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=kwargs.get("top_p", 0.75),
                    max_length=kwargs.get("max_length", 512),
                )
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text += " " + " ".join(decoded_outputs)

        return output_text.strip()


def dipper_paraphrase_json(input_path, output_path, lex=60, order=0, batch_size=4, **kwargs):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = DipperParaphraser()
    result_data = {"samples": {}}

    for checkpoint, content in data["samples"].items():
        model_texts = content.get("model_text", [])
        model_texts = model_texts[:1000]  
        paraphrased_texts = []

        for text in tqdm(model_texts, desc=f"Processing {checkpoint}"):
            rewritten = model.paraphrase(
                text,
                lex_div=lex,
                order_div=order,
                batch_size=batch_size,
                **kwargs
            )
            paraphrased_texts.append(rewritten)

        result_data["samples"][checkpoint] = {"model_text": paraphrased_texts}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIPPER Paraphraser Script")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--mode", choices=["dipper1", "dipper2"], required=True, help="Choose between dipper1 or dipper2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for paraphrasing")
    parser.add_argument("--top_p", type=float, default=0.75, help="Top-p value for nucleus sampling")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Choose mode
    if args.mode == "dipper1":
        lex, order = 60, 0
    elif args.mode == "dipper2":
        lex, order = 60, 20

    # Run paraphraser
    dipper_paraphrase_json(
        args.input,
        args.output,
        lex=lex,
        order=order,
        batch_size=args.batch_size,
        top_p=args.top_p
    )

