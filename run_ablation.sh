export VLLM_USE_V1=0
DATASET="nemotron"


# ----------

REWARD="combined_lsi"

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORTHAND="qwen7bins"
rm -rf /dev/shm/grpo_synthesis_models
GRPO_KWARGS="{\"model_name\": \"${MODEL_NAME}\", \"dataset_name\": \"/home/ubuntu/AcquisitionSynthesis/data/${DATASET}/train.parquet\"}"
source run_verl.sh "${MODEL_NAME}" "rewards/${REWARD}.py" "${MODEL_SHORTHAND}_${DATASET}_${REWARD}" "${DATASET}" "${REWARD}" "$GRPO_KWARGS"
notify "${MODEL_SHORTHAND}_${DATASET}_${REWARD} model is trained"

MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
MODEL_SHORTHAND="qwen14bins"
rm -rf /dev/shm/grpo_synthesis_models
GRPO_KWARGS="{\"model_name\": \"${MODEL_NAME}\", \"dataset_name\": \"/home/ubuntu/AcquisitionSynthesis/data/${DATASET}/train.parquet\"}"
source run_verl.sh "${MODEL_NAME}" "rewards/${REWARD}.py" "${MODEL_SHORTHAND}_${DATASET}_${REWARD}" "${DATASET}" "${REWARD}" "$GRPO_KWARGS"
notify "${MODEL_SHORTHAND}_${DATASET}_${REWARD} model is trained"

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORTHAND="llama8bins"
rm -rf /dev/shm/grpo_synthesis_models
GRPO_KWARGS="{\"model_name\": \"${MODEL_NAME}\", \"dataset_name\": \"/home/ubuntu/AcquisitionSynthesis/data/${DATASET}/train.parquet\"}"
source run_verl.sh "${MODEL_NAME}" "rewards/${REWARD}.py" "${MODEL_SHORTHAND}_${DATASET}_${REWARD}" "${DATASET}" "${REWARD}" "$GRPO_KWARGS"
notify "${MODEL_SHORTHAND}_${DATASET}_${REWARD} model is trained"



export CUDA_VISIBLE_DEVICES=0,1,2,3
source keep_alive.sh