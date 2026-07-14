import re
import json
import argparse
import torch
from vllm import LLM, SamplingParams

PROMETHEUS_PROMPT = lambda q, a, o: f"""###Task Description:
A question and a response will be given. Evaluate the correctness and accuracy of the response based on the reference answer. You MUST give a score between 1 and 5. Respond strictly with only: [RESULT] (score)

###Question:
{q}

###Response to Evaluate:
{a}

###Reference Answer (Score 5):
{o}

###Score Rubrics:
[Is the response correct and accurate?]
Score 1: The response is completely incorrect or irrelevant.
Score 2: The response is mostly incorrect with minor correct elements.
Score 3: The response is partially correct but lacks accuracy or completeness.
Score 4: The response is mostly correct with minor errors.
Score 5: The response is completely correct and accurate.

###Feedback:"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate_in", required=True)
    parser.add_argument("--out", default="hpo_judge_results.jsonl")
    args = parser.parse_args()

    with open(args.intermediate_in) as f:
        data = json.load(f)

    n_gpus = torch.cuda.device_count()
    judge_llm = LLM('prometheus-eval/prometheus-7b-v2.0', tensor_parallel_size=n_gpus, gpu_memory_utilization=0.7)
    judge_params = SamplingParams(temperature=0.0, max_tokens=256)

    judge_prompts = [PROMETHEUS_PROMPT(q, a, o) for q, a, o in zip(data["questions"], data["outputs"], data["answers"])]
    judge_outputs = judge_llm.generate(judge_prompts, judge_params)
    judge_outputs = [o.outputs[0].text.strip() for o in judge_outputs]

    def parse_score(text):
        match = re.search(r'\[RESULT\]\s*(\d)', text)
        return int(match.group(1)) if match else 1.0

    judge_scores = [parse_score(o) for o in judge_outputs]
    judge_acc = [1.0 if s >= 4.0 else 0.0 for s in judge_scores]

    result = {
        "trial_name": data["trial_name"],
        "model_path": data["model_path"],
        "embed_sim": data.get("embed_sim"),
        "rouge_l": data["rouge_l"],
        "judge": sum(judge_acc) / len(judge_acc) * 100,
        "n": data["n"],
    }
    with open(args.out, "a+") as f:
        f.write(json.dumps(result) + "\n")
    print(result)