"""
Build a cross-lingual variant of a (question, answer, reasoning) dataset:
the reasoning is translated into a language DIFFERENT from the question's
language, using Qwen2.5-32B-Instruct. Question and answer are left untouched.
"""

import argparse
import re

import pandas as pd
import torch
from langdetect import DetectorFactory, detect_langs
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

DetectorFactory.seed = 0

LANGUAGES = ["English", "Spanish", "French", "Italian", "Portuguese", "Arabic"]
ISO_TO_LANGUAGE = {"en": "English", "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese", "ar": "Arabic"}

TRANSLATE_PROMPT = lambda reasoning, source_language, target_language: f"""Translate the following reasoning from {source_language} to {target_language}.

Rules:
- Preserve the exact meaning and step-by-step structure of the reasoning.
- Keep all mathematical notation, LaTeX commands (e.g. \\frac, \\boxed), numbers, variable names, and code exactly as-is — do not translate them.
- Keep any HTML-like tags (e.g. <think>, </think>, <reasoning>, </reasoning>) exactly as-is, in English — do not translate or remove them.
- Output ONLY the translated reasoning, wrapped in <translation> </translation> tags. Do not add any commentary before or after.

<reasoning>
{reasoning}
</reasoning>"""


def detect_language(text):
    """Restrict langdetect to our 6 known languages so it can't wander off
    into a lookalike (e.g. Catalan for Spanish, Somali for Portuguese)."""
    try:
        for candidate in detect_langs(text):
            if candidate.lang in ISO_TO_LANGUAGE:
                return ISO_TO_LANGUAGE[candidate.lang]
    except Exception:
        pass
    return "English"


def assign_target_languages(source_languages):
    """Deterministic round-robin over LANGUAGES, skipped forward on collision
    with the source language, to keep the target distribution near-even."""
    targets = []
    cursor = 0
    for source in source_languages:
        target = LANGUAGES[cursor % len(LANGUAGES)]
        if target == source:
            cursor += 1
            target = LANGUAGES[cursor % len(LANGUAGES)]
        targets.append(target)
        cursor += 1
    return targets


def strip_wrapper_tags(text, tags=("translation", "reasoning")):
    """Strip leading/trailing tags the model echoes from the prompt (e.g. a
    dangling <translation> when generation got cut off before the closing
    tag, or a stray <reasoning> wrapper). Repeats until stable since these
    can stack, e.g. '<translation><reasoning> ... '."""
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for tag in tags:
            new_text = re.sub(rf"^</?{tag}>\s*", "", text, flags=re.IGNORECASE)
            new_text = re.sub(rf"\s*</?{tag}>$", "", new_text, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text.strip()
                changed = True
    return text


def parse_translation(text):
    match = re.search(r"<translation>(.*?)</translation>", text, re.DOTALL | re.IGNORECASE)
    translation = match.group(1).strip() if match else text.strip()
    return strip_wrapper_tags(translation)


def main():
    parser = argparse.ArgumentParser(description="Translate reasoning into a language different from the question.")
    parser.add_argument("--input_file", default="/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_nemotron_combined.parquet")
    parser.add_argument("--output_file", default="/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_nemotron_combined_crosslingual.parquet")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--max_model_len", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for a quick test run).")
    args = parser.parse_args()

    df = pd.read_parquet(args.input_file, engine="pyarrow")
    if args.limit is not None:
        df = df.iloc[: args.limit].reset_index(drop=True)
    df = df.dropna(subset=["question", "answer", "reasoning"]).reset_index(drop=True)

    print(f"Loaded {len(df)} rows from {args.input_file}")

    print("Detecting question language...")
    source_languages = [detect_language(q) for q in tqdm(df["question"])]
    target_languages = assign_target_languages(source_languages)

    print("Source language distribution:", pd.Series(source_languages).value_counts().to_dict())
    print("Target language distribution:", pd.Series(target_languages).value_counts().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": TRANSLATE_PROMPT(reasoning, source, target)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for reasoning, source, target in zip(df["reasoning"], source_languages, target_languages)
    ]

    print(f"[vllm] Loading translation model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    print(f"[vllm] Translating {len(prompts)} reasoning traces...")
    outputs = llm.generate(prompts, sampling_params)
    translated_reasoning = [parse_translation(output.outputs[0].text) for output in outputs]

    del llm
    torch.cuda.empty_cache()

    out_df = pd.DataFrame({
        "question": df["question"],
        "answer": df["answer"],
        "reasoning": translated_reasoning,
    })
    out_df.to_parquet(args.output_file)
    print(f"Saved {len(out_df)} cross-lingual rows to {args.output_file}")


if __name__ == "__main__":
    main()
