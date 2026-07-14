import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

BASE = "/home/ubuntu/AcquisitionSynthesis"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/dev/shm/hpo_sft_models"
RESULTS_FILE = os.path.join(BASE, "evaluation", "hpo_results.jsonl")
LOG_DIR = os.path.join(BASE, "evaluation", "hpo_logs")
os.makedirs(LOG_DIR, exist_ok=True)

DATASETS = {
    "semreasoning": os.path.join(BASE, "training_data", "qwen7bins_nemotron_stem_semreasoning_clean1000.parquet"),
    "hlrep": os.path.join(BASE, "training_data", "qwen7bins_nemotron_stem_hlrep_clean1000.parquet"),
    "confidence": os.path.join(BASE, "training_data", "qwen7bins_nemotron_stem_confidence_clean1000.parquet"),
    "answerdiff": os.path.join(BASE, "training_data", "qwen7bins_nemotron_stem_answerdiff_clean1000.parquet"),
    "dataenvgym": os.path.join(BASE, "training_data", "DataEnvGym_nemotron_stem_qwen7bins_clean1000.parquet"),
}

# lr, epochs, lora_r -- includes the current production defaults (lr=2e-4, epochs=5, r=16) as a control trial
GRID = [
    (2e-4, 5, 16),
    (2e-4, 1, 8),
    (1e-4, 2, 16),
    (1e-4, 3, 32),
    (5e-5, 1, 32),
    (5e-5, 2, 8),
    (5e-5, 3, 16),
    (2e-5, 2, 32),
    (2e-5, 3, 8),
    (2e-5, 5, 16),
]

GPU_GROUPS = [
    {"gpus": "0,1,2,3", "port": "5142"},
    {"gpus": "4,5,6,7", "port": "5143"},
]

trials = []
for dataset_name, file_path in DATASETS.items():
    for lr, epochs, r in GRID:
        trials.append({
            "dataset_name": dataset_name,
            "file_path": file_path,
            "lr": lr,
            "epochs": epochs,
            "r": r,
            "alpha": r * 2,
            "trial_name": f"hpo_{dataset_name}_lr{lr}_ep{epochs}_r{r}",
        })


def run(cmd, env, log_path):
    with open(log_path, "a+") as f:
        f.write(f"\n\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} ==== {' '.join(cmd)}\n")
        f.flush()
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=os.path.join(BASE, "evaluation"))
    return result.returncode


def run_trial(trial, gpu_group):
    name = trial["trial_name"]
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    model_path = os.path.join(OUTPUT_DIR, name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_group["gpus"]
    for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE']:
        env.pop(key, None)

    nproc = len(gpu_group["gpus"].split(","))

    print(f"[{name}] training...")
    rc = run([
        "torchrun", f"--nproc_per_node={nproc}", f"--master_port={gpu_group['port']}", "sft.py",
        "--model_name", MODEL_NAME,
        "--file_name", trial["file_path"],
        "--num_epochs", str(trial["epochs"]),
        "--learning_rate", str(trial["lr"]),
        "--lora_r", str(trial["r"]),
        "--lora_alpha", str(trial["alpha"]),
        "--student_name", name,
        "--output_dir", OUTPUT_DIR,
        "--num_data", "1000",
    ], env, log_path)
    if rc != 0:
        print(f"[{name}] TRAIN FAILED (rc={rc}), see {log_path}")
        shutil.rmtree(model_path, ignore_errors=True)
        return

    print(f"[{name}] merging...")
    rc = run([
        "python", "merge.py",
        "--model_path", model_path,
        "--base_model", MODEL_NAME,
        "--skip_push",
    ], env, log_path)
    if rc != 0:
        print(f"[{name}] MERGE FAILED (rc={rc}), see {log_path}")
        shutil.rmtree(model_path, ignore_errors=True)
        return

    intermediate_path = os.path.join(LOG_DIR, f"{name}.intermediate.json")

    print(f"[{name}] generating...")
    rc = run([
        "python", "hpo_generate.py",
        "--model_path", model_path,
        "--trial_name", name,
        "--intermediate_out", intermediate_path,
    ], env, log_path)
    if rc != 0:
        print(f"[{name}] GENERATE FAILED (rc={rc}), see {log_path}")
        shutil.rmtree(model_path, ignore_errors=True)
        return

    print(f"[{name}] embedding...")
    rc = run([
        "python", "hpo_embed.py",
        "--intermediate_in", intermediate_path,
        "--out", RESULTS_FILE,
    ], env, log_path)
    if rc != 0:
        print(f"[{name}] EMBED FAILED (rc={rc}), see {log_path}")

    os.remove(intermediate_path) if os.path.exists(intermediate_path) else None
    shutil.rmtree(model_path, ignore_errors=True)
    print(f"[{name}] done.")


def worker(gpu_group, trial_list):
    for trial in trial_list:
        run_trial(trial, gpu_group)


if __name__ == "__main__":
    lane_a = trials[0::2]
    lane_b = trials[1::2]

    print(f"Running {len(trials)} trials across 2 lanes ({len(lane_a)} / {len(lane_b)})")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(worker, GPU_GROUPS[0], lane_a)
        f2 = ex.submit(worker, GPU_GROUPS[1], lane_b)
        f1.result()
        f2.result()

    print("HPO sweep complete.")
