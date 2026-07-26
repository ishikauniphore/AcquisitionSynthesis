export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_NAMES=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")

cd evaluation

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    py model_inference.py --model ${MODEL_NAME} --english_reasoning "on"
    py model_embed.py --model ${MODEL_NAME}
    py model_rouge.py --model ${MODEL_NAME}
    py model_laj.py --model ${MODEL_NAME}
    # notify "english reasoning evals for ${MODEL_NAME}"
done
notify "DONE WITH ENG REASONING"

MODEL_NAMES=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    source run_eval.sh "${MODEL_NAME}"
done
notify "round 2 of base"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    source run_eval.sh "${MODEL_NAME}"
done
notify "round 3 of base"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    source run_eval.sh "${MODEL_NAME}"
done
notify "round 4 of base"

cd ..
source keep_alive.sh