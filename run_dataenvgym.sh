DATASET="nemotron_stem"

cd generating_data
py baseline_synthesis_DataEnvGym.py \
    --data "${DATASET}" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen7bins.parquet"
py baseline_synthesis_DataEnvGym.py \
    --data "${DATASET}" \
    --model "Qwen/Qwen2.5-14B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen14bins.parquet"
py baseline_synthesis_DataEnvGym.py \
    --data "${DATASET}" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_llama8bins.parquet"
cd ..



cd /home/ubuntu/AcquisitionSynthesis/evaluation
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen7bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_qwen7bins" --base_model "Qwen/Qwen2.5-7B-Instruct"
rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_qwen7bins

torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen14bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_qwen14bins" --base_model "Qwen/Qwen2.5-14B-Instruct"
rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_qwen14bins

torchrun --nproc_per_node=4 sft.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_llama8bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"
rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_llama8bins



source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_qwen7bins"
source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_qwen14bins"
source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_llama8bins"
cd ..

notify "dataenv gym done"