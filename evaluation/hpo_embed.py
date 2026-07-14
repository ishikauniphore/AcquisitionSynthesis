import os
os.environ["VLLM_USE_V1"] = "0"  # required for the embed task
import json
import argparse
import torch
from vllm import LLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate_in", required=True)
    parser.add_argument("--out", default="hpo_results.jsonl")
    args = parser.parse_args()

    with open(args.intermediate_in) as f:
        data = json.load(f)

    n_gpus = torch.cuda.device_count()
    embed_model = LLM('Qwen/Qwen3-Embedding-8B', task="embed", tensor_parallel_size=n_gpus, gpu_memory_utilization=0.7)
    pred_embs = torch.tensor([o.outputs.embedding for o in embed_model.embed(data["outputs"])])
    ref_embs = torch.tensor([o.outputs.embedding for o in embed_model.embed(data["answers"])])
    embed_sim = (pred_embs @ ref_embs.T).diag().tolist()
    embed_sim_acc = [1.0 if s > 0.9 else 0.0 for s in embed_sim]

    result = {
        "trial_name": data["trial_name"],
        "model_path": data["model_path"],
        "embed_sim": sum(embed_sim_acc) / len(embed_sim_acc) * 100,
        "rouge_l": data["rouge_l"],
        "n": data["n"],
    }
    with open(args.out, "a+") as f:
        f.write(json.dumps(result) + "\n")
    print(result)

    data["embed_sim"] = result["embed_sim"]
    with open(args.intermediate_in, "w") as f:
        json.dump(data, f)
