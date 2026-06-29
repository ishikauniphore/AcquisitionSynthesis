DATASET="nemotron_stem"

### STEP 2: Dataset generations
export CUDA_VISIBLE_DEVICES=0,1,2,3
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "training_data/Original_nemotron_stem_qwen7bins_10k.parquet" \
    --size 10000 --k 4

# ### STEP 3: Student evaluation
cd evaluation
rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "training_data/Original_nemotron_stem_qwen7bins_10k.parquet"
    --n 10000
py merge.py --model_path "/dev/shm/sft_models/Original_nemotron_stem_qwen7bins_10k"

source run_eval.sh "ishikauniphore/student_Original_nemotron_stem_qwen7bins_10k"

rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "training_data/Original_nemotron_stem_qwen7bins_5k.parquet"
    --n 5000
py merge.py --model_path "/dev/shm/sft_models/Original_nemotron_stem_qwen7bins_5k"

source run_eval.sh "ishikauniphore/student_Original_nemotron_stem_qwen7bins_5k"
cd ..

notify "ORIGINAL LEARNING CURVE DONE"