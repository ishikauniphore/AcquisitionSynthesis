DATASET="nemotron_stem"

### STEP 2: Formatting
py generating_data/baseline_selection_random.py --data "${DATASET}" --size 1000 --file "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet"

### STEP 3: Student evaluation
cd evaluation
rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 1000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet"
py merge.py --model_path "/dev/shm/sft_models/selected_${DATASET}" --base_model "Qwen/Qwen2.5-7B-Instruct"
source run_eval.sh "ishikauniphore/student_selected_${DATASET}"

cd ..
notify "SELECTION DONE"

source keep_alive.sh


