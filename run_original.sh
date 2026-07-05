DATASET="nemotron_math"

### STEP 2: Dataset generations
export CUDA_VISIBLE_DEVICES=0,1,2,3
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "training_data/Original_${DATASET}_qwen7bins.parquet" \
    --size 1000 --k 4
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --answer_model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --output_file "training_data/Original_${DATASET}_llama8bins.parquet" \
    --size 1000 --k 4
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "Qwen/Qwen2.5-14B-Instruct" \
    --answer_model_name "Qwen/Qwen2.5-14B-Instruct" \
    --output_file "training_data/Original_${DATASET}_qwen14bins.parquet" \
    --size 1000 --k 4
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "Qwen/Qwen2.5-14B-Instruct" \
    --answer_model_name "Qwen/Qwen2.5-14B-Instruct" \
    --output_file "training_data/Original_${DATASET}_qwen14bins.parquet" \
    --questions_file "training_data/questions.parquet" \
    --size 1000 --k 4

# ### STEP 3: Student evaluation
cd evaluation
rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/Original_${DATASET}_qwen7bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/Original_${DATASET}_qwen7bins" --base_model "Qwen/Qwen2.5-7B-Instruct"

rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/Original_${DATASET}_qwen14bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/Original_${DATASET}_qwen14bins" --base_model "Qwen/Qwen2.5-14B-Instruct"

rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=4 sft.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/Original_${DATASET}_llama8bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/Original_${DATASET}_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"

source run_eval.sh "ishikauniphore/student_Original_${DATASET}_qwen7bins"
source run_eval.sh "ishikauniphore/student_Original_${DATASET}_qwen14bins"
source run_eval.sh "ishikauniphore/student_Original_${DATASET}_llama8bins"


cd ..
notify "ORIGINAL DONE"

source keep_alive.sh
