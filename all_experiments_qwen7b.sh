DATASET="nemotron"

# combined
export CUDA_VISIBLE_DEVICES=4,5,6,7
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "ishikauniphore/generator_qwenREAL7bins_nemotron_combined" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/qwenREAL7bins_${DATASET}_combined.parquet" \
    --size 5000 --k 4
cd evaluation
torchrun --nproc_per_node=4 --master_port=5142 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 5000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/qwenREAL7bins_${DATASET}_combined.parquet"
py merge.py --model_path "/dev/shm/sft_models/qwenREAL7bins_${DATASET}_combined" --base_model "Qwen/Qwen2.5-7B-Instruct"
rm -rf "/dev/shm/sft_models/qwenREAL7bins_${DATASET}_combined"
source run_eval.sh "ishikauniphore/student_qwenREAL7bins_${DATASET}_combined"
notify "combined evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis

### English Reasoning
# cd /home/ubuntu/AcquisitionSynthesis
# source run_englishreasoning.sh


cd /home/ubuntu/AcquisitionSynthesis

notify "now just keeping it alive"
source keep_alive.sh