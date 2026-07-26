DATASET="nemotron"

### Selection
# py generating_data/baseline_selection_random.py --data "${DATASET}" --size 5000 --file "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet"
# cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_qwen14bins_${DATASET}.parquet"
# cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGT_qwen14bins_${DATASET}.parquet"

# Generated Answers
# py generating_data/data_gen_cluster.py \
#     --dataset_name "${DATASET}" \
#     --acquisition_model_name "Qwen/Qwen2.5-14B-Instruct" \
#     --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
#     --output_file "training_data/SelectedGEN_${DATASET}_qwen14bins.parquet" \
#     --questions_file "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_qwen14bins_${DATASET}.parquet" \
#     --size 5000 --k 4
cd evaluation
torchrun --nproc_per_node=4 --master_port=5141 sft.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" --num_data 5000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_${DATASET}_qwen14bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/SelectedGEN_${DATASET}_qwen14bins" --base_model "Qwen/Qwen2.5-14B-Instruct"
rm -rf /dev/shm/sft_models/SelectedGEN_${DATASET}_qwen14bins
source run_eval.sh "ishikauniphore/student_SelectedGEN_${DATASET}_qwen14bins"
cd /home/ubuntu/AcquisitionSynthesis

# GT Answers
cd evaluation
torchrun --nproc_per_node=4 --master_port=5141 sft.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" --num_data 5000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGT_qwen14bins_${DATASET}.parquet"
py merge.py --model_path "/dev/shm/sft_models/SelectedGT_qwen14bins_${DATASET}" --base_model "Qwen/Qwen2.5-14B-Instruct"
rm -rf /dev/shm/sft_models/SelectedGT_qwen14bins_${DATASET}
source run_eval.sh "ishikauniphore/student_SelectedGT_qwen14bins_${DATASET}"
notify "selected evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis

### DataEnvGym
py generating_data/baseline_synthesis_DataEnvGym.py \
    --data "${DATASET}" \
    --student_model "Qwen/Qwen2.5-14B-Instruct" \
    --teacher_model "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen14bins.parquet" \
    --size 5000
cd evaluation
torchrun --nproc_per_node=4 --master_port=5141 sft.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" --num_data 5000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen14bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_qwen14bins" --base_model "Qwen/Qwen2.5-14B-Instruct"
rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_qwen14bins
source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_qwen14bins"
notify "dataenvgym evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis




### Untrained Data Generator
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "Qwen/Qwen2.5-14B-Instruct" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "training_data/Original_${DATASET}_qwen14bins.parquet" \
    --size 5000 --k 4
cd evaluation
torchrun --nproc_per_node=4 --master_port=5141 sft.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" --num_data 5000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/Original_${DATASET}_qwen14bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/Original_${DATASET}_qwen14bins" --base_model "Qwen/Qwen2.5-14B-Instruct"
rm -rf /dev/shm/sft_models/Original_${DATASET}_qwen14bins
source run_eval.sh "ishikauniphore/student_Original_${DATASET}_qwen14bins"
notify "untrained data generator evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis



### English Reasoning
cd /home/ubuntu/AcquisitionSynthesis
source run_englishreasoning.sh


cd /home/ubuntu/AcquisitionSynthesis

notify "now just keeping it alive"
source keep_alive.sh