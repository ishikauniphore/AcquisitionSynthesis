DATASET="nemotron"

# ### Base
# cd evaluation
# source run_eval.sh "meta-llama/Llama-3.1-8B-Instruct"
# cd /home/ubuntu/AcquisitionSynthesis

# ### Selection
# py generating_data/baseline_selection_random.py --data "${DATASET}" --size 5000 --file "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet"
# cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_llama8bins_${DATASET}.parquet"
# cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGT_llama8bins_${DATASET}.parquet"

# # Generated Answers
# py generating_data/data_gen_cluster.py \
#     --dataset_name "${DATASET}" \
#     --acquisition_model_name "meta-llama/Llama-3.1-8B-Instruct" \
#     --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
#     --output_file "training_data/SelectedGEN_${DATASET}_llama8bins.parquet" \
#     --questions_file "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_llama8bins_${DATASET}.parquet" \
#     --size 5000 --k 4
# cd evaluation
# torchrun --nproc_per_node=4 --master_port=5141 sft.py \
#     --model_name "meta-llama/Llama-3.1-8B-Instruct" --num_data 5000 \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_${DATASET}_llama8bins.parquet"
# py merge.py --model_path "/dev/shm/sft_models/SelectedGEN_${DATASET}_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"
# rm -rf /dev/shm/sft_models/SelectedGEN_${DATASET}_llama8bins
# source run_eval.sh "ishikauniphore/student_SelectedGEN_${DATASET}_llama8bins"
# cd /home/ubuntu/AcquisitionSynthesis

# # GT Answers
# cd evaluation
# torchrun --nproc_per_node=4 --master_port=5141 sft.py \
#     --model_name "meta-llama/Llama-3.1-8B-Instruct" --num_data 5000 \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGT_llama8bins_${DATASET}.parquet"
# py merge.py --model_path "/dev/shm/sft_models/SelectedGT_llama8bins_${DATASET}" --base_model "meta-llama/Llama-3.1-8B-Instruct"
# rm -rf /dev/shm/sft_models/SelectedGT_llama8bins_${DATASET}
# source run_eval.sh "ishikauniphore/student_SelectedGT_llama8bins_${DATASET}"
# notify "selected evals done!!!!"
# rm -rf /dev/shm/hf_cache/hub/*
# cd /home/ubuntu/AcquisitionSynthesis

### DataEnvGym
py generating_data/baseline_synthesis_DataEnvGym.py \
    --data "${DATASET}" \
    --student_model "meta-llama/Llama-3.1-8B-Instruct" \
    --teacher_model "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_llama8bins.parquet" \
    --size 5000
cd evaluation
torchrun --nproc_per_node=4 --master_port=5141 sft.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" --num_data 5000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_llama8bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"
rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_llama8bins
source run_eval.sh "ishikauniphore/student_DataEnvGym_${DATASET}_llama8bins"
notify "dataenvgym evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis




### Untrained Data Generator
# py generating_data/data_gen_cluster.py \
#     --dataset_name "${DATASET}" \
#     --acquisition_model_name "meta-llama/Llama-3.1-8B-Instruct" \
#     --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
#     --output_file "training_data/Original_${DATASET}_llama8bins.parquet" \
#     --size 5000 --k 4
# cd evaluation
# torchrun --nproc_per_node=4 --master_port=5141 sft.py \
#     --model_name "meta-llama/Llama-3.1-8B-Instruct" --num_data 5000 \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/Original_${DATASET}_llama8bins.parquet"
# py merge.py --model_path "/dev/shm/sft_models/Original_${DATASET}_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"
# rm -rf /dev/shm/sft_models/Original_${DATASET}_llama8bins
# source run_eval.sh "ishikauniphore/student_Original_${DATASET}_llama8bins"
# notify "untrained data generator evals done!!!!"
# rm -rf /dev/shm/hf_cache/hub/*
# cd /home/ubuntu/AcquisitionSynthesis





### TANDEM
# AnswerDiff
# py generating_data/data_gen_cluster.py \
#     --dataset_name "${DATASET}" \
#     --acquisition_model_name "${HF_USERNAME}/generator_llama8bins_${DATASET}_answerdiff" \
#     --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
#     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/llama8bins_${DATASET}_answerdiff.parquet" \
#     --size 5000 --k 4
# cd evaluation
# torchrun --nproc_per_node=4 --master_port=5142 sft.py \
#     --model_name "meta-llama/Llama-3.1-8B-Instruct" --num_data 5000 \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/llama8bins_${DATASET}_answerdiff.parquet"
# py merge.py --model_path "/dev/shm/sft_models/llama8bins_${DATASET}_answerdiff" --base_model "meta-llama/Llama-3.1-8B-Instruct"
# rm -rf "/dev/shm/sft_models/llama8bins_${DATASET}_answerdiff"
# source run_eval.sh "ishikauniphore/student_llama8bins_${DATASET}_answerdiff"
# notify "answerdiff evals done!!!!"
# rm -rf /dev/shm/hf_cache/hub/*
# cd /home/ubuntu/AcquisitionSynthesis


# # HLRep
# py generating_data/data_gen_cluster.py \
#     --dataset_name "${DATASET}" \
#     --acquisition_model_name "${HF_USERNAME}/generator_llama8bins_${DATASET}_hlrep" \
#     --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
#     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/llama8bins_${DATASET}_hlrep.parquet" \
#     --size 5000 --k 4
# cd evaluation
# torchrun --nproc_per_node=4 --master_port=5142 sft.py \
#     --model_name "meta-llama/Llama-3.1-8B-Instruct" --num_data 5000 \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/llama8bins_${DATASET}_hlrep.parquet"
# py merge.py --model_path "/dev/shm/sft_models/llama8bins_${DATASET}_hlrep" --base_model "meta-llama/Llama-3.1-8B-Instruct"
# rm -rf "/dev/shm/sft_models/llama8bins_${DATASET}_hlrep"
# source run_eval.sh "ishikauniphore/student_llama8bins_${DATASET}_hlrep"
# notify "hlrep evals done!!!!"
# rm -rf /dev/shm/hf_cache/hub/*
# cd /home/ubuntu/AcquisitionSynthesis

### English Reasoning
# cd /home/ubuntu/AcquisitionSynthesis
# source run_englishreasoning.sh


# cd /home/ubuntu/AcquisitionSynthesis

# notify "now just keeping it alive"
# source keep_alive.sh