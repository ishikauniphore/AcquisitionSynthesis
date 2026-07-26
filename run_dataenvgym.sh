export CUDA_VISIBLE_DEVICES=4,5,6,7
DATASET="nemotron"

# cd generating_data
# # py baseline_synthesis_DataEnvGym.py \
# #     --data "${DATASET}" \
# #     --model "Qwen/Qwen2.5-7B-Instruct" \
# #     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen7bins.parquet"
# # py baseline_synthesis_DataEnvGym.py \
# #     --data "${DATASET}" \
# #     --model "Qwen/Qwen2.5-14B-Instruct" \
# #     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen14bins.parquet"
# # py baseline_synthesis_DataEnvGym.py \
# #     --data "${DATASET}" \
# #     --model "meta-llama/Llama-3.1-8B-Instruct" \
# #     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_llama8bins.parquet"
# cd ..



# cd /home/ubuntu/AcquisitionSynthesis/evaluation
# torchrun --nproc_per_node=4 --master_port=5141 sft.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen7bins.parquet"
# py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_qwen7bins" --base_model "Qwen/Qwen2.5-7B-Instruct"
# rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_qwen7bins

# # torchrun --nproc_per_node=1 sft.py \
# #     --model_name "Qwen/Qwen2.5-14B-Instruct" \
# #     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen14bins.parquet"
# # py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_qwen14bins" --base_model "Qwen/Qwen2.5-14B-Instruct"
# # rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_qwen14bins

# # torchrun --nproc_per_node=1 sft.py \
# #     --model_name "meta-llama/Llama-3.1-8B-Instruct" \
# #     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_llama8bins.parquet"
# # py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"
# # rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_llama8bins



# source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_qwen7bins"
# # source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_qwen14bins"
# # source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_llama8bins"
# # cd ..

# notify "dataenv gym done"
# source keep_alive.sh





py generating_data/baseline_synthesis_DataEnvGym.py \
    --data "${DATASET}" \
    --student_model "Qwen/Qwen2.5-7B-Instruct" \
    --teacher_model "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen7bins.parquet" \
    --size 1000
cd evaluation
torchrun --nproc_per_node=4 --master_port=5141 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 1000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen7bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_qwen7bins" --base_model "Qwen/Qwen2.5-7B-Instruct"
rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_qwen7bins
source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_qwen7bins"
notify "dataenvgym evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis