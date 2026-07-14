# DATASET="nemotron_stem"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_SHORTHAND="llama8bins"
# REWARDS=("answerdiff" "confidence" "semreasoning" "hlrep")
# for REWARD in "${REWARDS[@]}"; do
#     ### STEP 2: Dataset generation
#     py generating_data/data_gen_cluster.py \
#         --dataset_name "${DATASET}" \
#         --acquisition_model_name "${HF_USERNAME}/generator_${MODEL_SHORTHAND}_${DATASET}_${REWARD}" \
#         --answer_model_name "${MODEL_NAME}" \
#         --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/${MODEL_SHORTHAND}_${DATASET}_${REWARD}.parquet" \
#     --size 1000 --k 4

#     ### STEP 3: Student evaluation
#     cd evaluation
#     rm -rf /dev/shm/sft_models/
#     torchrun --nproc_per_node=4 --master_port=29500 sft.py \
#         --model_name "${MODEL_NAME}" \
#         --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/${MODEL_SHORTHAND}_${DATASET}_${REWARD}.parquet"
#     py merge.py --model_path "/dev/shm/sft_models/${MODEL_SHORTHAND}_${DATASET}_${REWARD}" --base_model "${MODEL_NAME}"

#     source run_eval.sh "ishikauniphore/student_${MODEL_SHORTHAND}_${DATASET}_${REWARD}"
#     cd ..

#     notify "experiment done!!!! 0_0 ${MODEL_SHORTHAND}_${DATASET}_${REWARD}"
# done
# source keep_alive.sh



DATASET="nemotron_stem"
export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
MODEL_SHORTHAND="qwen14bins"
REWARDS=("semreasoning" "hlrep")
for REWARD in "${REWARDS[@]}"; do
    ### STEP 2: Dataset generation
    py generating_data/data_gen_cluster.py \
        --dataset_name "${DATASET}" \
        --acquisition_model_name "${HF_USERNAME}/generator_${MODEL_SHORTHAND}_${DATASET}_${REWARD}" \
        --answer_model_name "${MODEL_NAME}" \
        --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/${MODEL_SHORTHAND}_${DATASET}_${REWARD}.parquet" \
    --size 1000 --k 4

    ### STEP 3: Student evaluation
    cd evaluation
    rm -rf /dev/shm/sft_models/
    torchrun --nproc_per_node=4 --master_port=29501 sft.py \
        --model_name "${MODEL_NAME}" \
        --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/${MODEL_SHORTHAND}_${DATASET}_${REWARD}.parquet"
    py merge.py --model_path "/dev/shm/sft_models/${MODEL_SHORTHAND}_${DATASET}_${REWARD}" --base_model "${MODEL_NAME}"

    source run_eval.sh "ishikauniphore/student_${MODEL_SHORTHAND}_${DATASET}_${REWARD}"
    cd ..

    notify "experiment done!!!! 0_0 ${MODEL_SHORTHAND}_${DATASET}_${REWARD}"
done
source keep_alive.sh