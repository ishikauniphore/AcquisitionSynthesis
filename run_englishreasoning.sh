MODEL_NAMES=("Qwen/Qwen2.5-14B-Instruct")

cd evaluation
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    py model_inference.py --model ${MODEL_NAME} --english_reasoning "on"
    py model_embed.py --model ${MODEL_NAME}
    py model_rouge.py --model ${MODEL_NAME}
    py model_laj.py --model ${MODEL_NAME}
    notify "english reasoning evals for ${MODEL_NAME}"
done

cd ..