import os
import sys
import json
import argparse
import torch
import pandas as pd
from vllm import LLM, SamplingParams
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import get_prompt_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", default="/home/ubuntu/AcquisitionSynthesis/data/nemotron_stem/test.parquet")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--trial_name", required=True)
    parser.add_argument("--intermediate_out", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.test_file)
    questions = list(df.apply(lambda row: row['extra_info']['grounding_question'], axis=1))[:args.n]
    answers = list(df.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))[:args.n]
    answers = [a.split("\\boxed{")[-1].split("}")[0].strip() for a in answers]

    prompt_template = get_prompt_template("stem")
    prompts = [prompt_template(q) for q in questions]

    n_gpus = torch.cuda.device_count()
    llm = LLM(args.model_path, tensor_parallel_size=n_gpus, gpu_memory_utilization=0.7, trust_remote_code=True, enforce_eager=True, max_model_len=8096)
    sampling_params = SamplingParams(temperature=0.2, max_tokens=2048)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = [o.outputs[0].text.strip() for o in outputs]
    outputs = [o.split("\\boxed{")[-1].split("}")[0].strip() for o in outputs]
    outputs = [o[:1].upper() if len(o) > 2 and o[1] == ":" else o for o in outputs]

    rs = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = [rs.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(outputs, answers)]
    rouge_l_acc = [1.0 if s > 0.5 else 0.0 for s in rouge_l]

    with open(args.intermediate_out, "w") as f:
        json.dump({
            "trial_name": args.trial_name,
            "model_path": args.model_path,
            "questions": questions,
            "answers": answers,
            "outputs": outputs,
            "rouge_l": sum(rouge_l_acc) / len(rouge_l_acc) * 100,
            "n": len(outputs),
        }, f)
