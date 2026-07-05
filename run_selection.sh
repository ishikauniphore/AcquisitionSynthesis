DATASET="nemotron_stem"

### STEP 2: Formatting
py generating_data/baseline_selection_random.py --data "${DATASET}" --size 1000 --file "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet"

cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}_qwen7bins.parquet"
cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}_qwen14bins.parquet"
cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}_llama8bins.parquet"

### STEP 3: Student evaluation
cd evaluation
rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}_qwen7bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/selected_${DATASET}_qwen7bins" --base_model "Qwen/Qwen2.5-7B-Instruct"

rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}_qwen14bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/selected_${DATASET}_qwen14bins" --base_model "Qwen/Qwen2.5-14B-Instruct"

rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}_llama8bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/selected_${DATASET}_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"

source run_eval.sh "ishikauniphore/student_selected_${DATASET}_qwen7bins"
source run_eval.sh "ishikauniphore/student_selected_${DATASET}_qwen14bins"
source run_eval.sh "ishikauniphore/student_selected_${DATASET}_llama8bins"


cd ..
notify "SELECTION DONE"

source keep_alive.sh


