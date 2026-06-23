export VLLM_USE_V1=0
REWARD="mcot"
DATASET="nemotron_stem"



### STEP 1: Acquisition training
rm -rf /tmp/grpo_synthesis_models
GRPO_KWARGS="{\"model_name\": \"Qwen/Qwen2.5-7B-Instruct\", \"dataset_name\": \"/home/ubuntu/AcquisitionSynthesis/data/${DATASET}/train.parquet\"}"
source run_verl.sh "Qwen/Qwen2.5-3B-Instruct" "rewards/${REWARD}.py" "3bT-7bS-v3_${DATASET}_${REWARD}" "${DATASET}" "${REWARD}" "$GRPO_KWARGS"

notify "model is trained"


export CUDA_VISIBLE_DEVICES=0,1,2,3
# STEP 2: Dataset generations
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "${HF_USERNAME}/generator_3bT-7bS-v3_${DATASET}_${REWARD}" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "training_data/3bT-7bS-v3_${DATASET}_${REWARD}.parquet" \
    --size 1000 --k 4

# ### STEP 3: Student evaluation
cd evaluation
rm -rf /tmp/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/3bT-7bS-v3_${DATASET}_${REWARD}.parquet"
py merge.py --model_path "/tmp/sft_models/3bT-7bS-v3_${DATASET}_${REWARD}"

source run_eval.sh "ishikauniphore/student_3bT-7bS-v3_${DATASET}_${REWARD}"
cd ..
rm -rf ~/.cache/huggingface/hub/*ishikauniphore*

notify "experiment done!!!! 0_0"