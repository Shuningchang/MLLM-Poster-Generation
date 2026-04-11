import os
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Stable Diffusion prompts from reference posters and product images."
    )
    parser.add_argument("--csv_path", type=str, default="data/pairs.csv", help="Path to pairs.csv")
    parser.add_argument("--product_dir", type=str, default="data/product", help="Directory of product images")
    parser.add_argument("--ref_dir", type=str, default="data/ref", help="Directory of reference images")
    parser.add_argument("--output_json", type=str, default="outputs/prompts.json", help="Output JSON path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug_samples", type=int, default=5, help="Number of samples in debug mode")
    return parser.parse_args()


def build_instruction(product_title: str) -> str:
    return (
        "Task: Generate a Stable Diffusion prompt for a product advertisement poster.\n\n"
        "Step 1 - Analyze the FIRST image (reference poster):\n"
        "Note its color palette, background style, layout, lighting, and mood.\n\n"
        "Step 2 - Look at the SECOND image (product).\n\n"
        "Step 3 - Output a prompt that places this product in a poster "
        "with the SAME visual style as the reference poster.\n\n"
        "Rules:\n"
        "- Output ONE line only\n"
        "- Comma-separated short phrases (NOT full sentences)\n"
        "- 10-15 phrases total\n"
        "- Start with a brief product description (2-3 words max)\n"
        "- Then describe: poster style, dominant colors, background, lighting, composition\n"
        "- End with: high quality, commercial photography\n"
        "- Do NOT output anything except the prompt\n\n"
        "Example format:\n"
        "sleek USB enclosure, minimalist advertisement poster, silver and white tones, "
        "gradient background, centered product, soft studio lighting, bold sans-serif text, "
        "clean layout, premium feel, high quality, commercial photography\n\n"
        "Now generate the prompt for this product:\n"
        f"{product_title[:50]}\n\n"
        "Prompt:"
    )


def clean_prompt(text: str, product_title: str) -> str:
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    elif "assistant" in text:
        text = text.split("assistant")[-1]

    for prefix in ["Prompt:", "Output:", "Final prompt:", "SD prompt:"]:
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):]

    text = text.strip()

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = lines[0] if lines else text

    parts = [part.strip() for part in text.split(",") if part.strip()]
    seen = []
    seen_lower = set()

    for part in parts:
        if part.lower() not in seen_lower and len(part) < 60:
            seen.append(part)
            seen_lower.add(part.lower())

    if len(seen) < 5:
        short_title = " ".join(product_title.split()[:4])
        return (
            f"{short_title}, product advertisement poster, "
            "clean background, professional lighting, bold typography, "
            "high quality, commercial photography"
        )

    return ", ".join(seen[:15])


def load_model_and_processor(model_id: str):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    model.eval()
    return processor, model


def generate_prompt_for_pair(processor, model, product_path: str, ref_path: str, product_title: str) -> str:
    instruction = build_instruction(product_title)

    ref_image = Image.open(ref_path).convert("RGB")
    product_image = Image.open(product_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reference poster:"},
                {"type": "image", "image": ref_image},
                {"type": "text", "text": "Product image:"},
                {"type": "image", "image": product_image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[ref_image, product_image],
        return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, input_len:]
    result = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    return clean_prompt(result, product_title)


def validate_csv_columns(df: pd.DataFrame):
    required_columns = ["pair_id", "product_image", "ref_image", "product_title"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"pairs.csv is missing a required column:{col}")


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    product_dir = Path(args.product_dir)
    ref_dir = Path(args.ref_dir)
    output_json = Path(args.output_json)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    validate_csv_columns(df)

    if args.debug:
        df = df.head(args.debug_samples).reset_index(drop=True)

    print(f"Loaded {len(df)} entries")

    processor, model = load_model_and_processor(args.model_id)

    results = []

    for i, row in df.iterrows():
        pair_id = str(row["pair_id"]).zfill(4)
        product_path = product_dir / row["product_image"]
        ref_path = ref_dir / row["ref_image"]
        product_title = row["product_title"]

        print(f"\n[{i + 1}/{len(df)}] pair_id={pair_id}")

        if not product_path.exists():
            print(f" Skip: {product_path} not found")
            continue

        if not ref_path.exists():
            print(f"  Skip: {ref_path} not found")
            continue

        try:
            prompt = generate_prompt_for_pair(
                processor=processor,
                model=model,
                product_path=str(product_path),
                ref_path=str(ref_path),
                product_title=product_title,
            )
            print(f"  → {prompt}")
            results.append({
                "pair_id": pair_id,
                "generated_prompt": prompt
            })
        except Exception as e:
            print(f"  Error：{e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Output saved to {output_json}, total {len(results)} entries")


if __name__ == "__main__":
    main()
