DATASET="nemotron_stem"

### Selection
# py generating_data/baseline_selection_random.py --data "${DATASET}" --size 2000 --file "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet"
# cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_qwen7bins_${DATASET}.parquet"
# cp "/home/ubuntu/AcquisitionSynthesis/training_data/selected_${DATASET}.parquet" "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGT_qwen7bins_${DATASET}.parquet"
# # Generated Answers
# py generating_data/data_gen_cluster.py \
#     --dataset_name "${DATASET}" \
#     --acquisition_model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
#     --output_file "training_data/SelectedGEN_${DATASET}_qwen7bins.parquet" \
#     --questions_file "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_qwen7bins_${DATASET}.parquet" \
#     --size 2000 --k 4
# cd evaluation
# torchrun --nproc_per_node=4 --master_port=5141 sft.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 2000 \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGEN_${DATASET}_qwen7bins.parquet"
# py merge.py --model_path "/dev/shm/sft_models/SelectedGEN_${DATASET}_qwen7bins" --base_model "Qwen/Qwen2.5-7B-Instruct"
# rm -rf /dev/shm/sft_models/SelectedGEN_${DATASET}_qwen7bins
# source run_eval.sh "ishikauniphore/student_SelectedGEN_${DATASET}_qwen7bins"
# cd /home/ubuntu/AcquisitionSynthesis

# GT Answers
# cd evaluation
# torchrun --nproc_per_node=4 --master_port=5141 sft.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 2000 \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/SelectedGT_qwen7bins_${DATASET}.parquet"
# py merge.py --model_path "/dev/shm/sft_models/SelectedGT_qwen7bins_${DATASET}" --base_model "Qwen/Qwen2.5-7B-Instruct"
# rm -rf /dev/shm/sft_models/SelectedGT_qwen7bins_${DATASET}
# source run_eval.sh "ishikauniphore/student_SelectedGT_qwen7bins_nemotron_stem"
# cd /home/ubuntu/AcquisitionSynthesis

# notify "selected evals done!!!!"
# rm -rf /dev/shm/hf_cache/hub/*

# ### DataEnvGym
# py generating_data/baseline_synthesis_DataEnvGym.py \
#     --data "${DATASET}" \
#     --student_model "Qwen/Qwen2.5-7B-Instruct" \
#     --teacher_model "Qwen/Qwen2.5-32B-Instruct" \
#     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen7bins.parquet" \
#     --size 2000 --k 4
# cd evaluation
# torchrun --nproc_per_node=4 --master_port=5141 sft.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 2000 \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_${DATASET}_qwen7bins.parquet"
# py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_${DATASET}_qwen7bins" --base_model "Qwen/Qwen2.5-7B-Instruct"
# rm -rf /dev/shm/sft_models/DataEnvGym_${DATASET}_qwen7bins
# source run_eval.sh "ishikauniphore/student_DataEnvGym_nemotron_stem_qwen7bins"
# notify "dataenvgym evals done!!!!"
# rm -rf /dev/shm/hf_cache/hub/*
# cd /home/ubuntu/AcquisitionSynthesis


### Untrained Data Generator
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "training_data/Original_${DATASET}_qwen7bins.parquet" \
    --size 2000 --k 4
cd evaluation
torchrun --nproc_per_node=4 --master_port=5141 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 2000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/Original_${DATASET}_qwen7bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/Original_${DATASET}_qwen7bins" --base_model "Qwen/Qwen2.5-7B-Instruct"
rm -rf /dev/shm/sft_models/Original_${DATASET}_qwen7bins
source run_eval.sh "ishikauniphore/student_Original_nemotron_stem_qwen7bins"
notify "untrained data generator evals done!!!!"
cd /home/ubuntu/AcquisitionSynthesis
rm -rf /dev/shm/hf_cache/hub/*

### TANDEM
# AnswerDiff
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "${HF_USERNAME}/generator_qwen7bins_${DATASET}_answerdiff" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_${DATASET}_answerdiff.parquet" \
    --size 2000 --k 4
cd evaluation
torchrun --nproc_per_node=4 --master_port=5142 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 2000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_${DATASET}_answerdiff.parquet"
py merge.py --model_path "/dev/shm/sft_models/qwen7bins_${DATASET}_answerdiff" --base_model "Qwen/Qwen2.5-7B-Instruct"
rm -rf "/dev/shm/sft_models/qwen7bins_${DATASET}_answerdiff"
source run_eval.sh "ishikauniphore/student_qwen7bins_nemotron_stem_answerdiff"
notify "answerdiff evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis


# Confidence
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "${HF_USERNAME}/generator_qwen7bins_${DATASET}_confidence" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_${DATASET}_confidence.parquet" \
    --size 2000 --k 4
cd evaluation
torchrun --nproc_per_node=4 --master_port=5142 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 2000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_${DATASET}_confidence.parquet"
py merge.py --model_path "/dev/shm/sft_models/qwen7bins_${DATASET}_confidence" --base_model "Qwen/Qwen2.5-7B-Instruct"
rm -rf "/dev/shm/sft_models/qwen7bins_${DATASET}_confidence"
source run_eval.sh "ishikauniphore/student_qwen7bins_nemotron_stem_confidence"
notify "confidence evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis


# SemanticReasoning
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "${HF_USERNAME}/generator_qwen7bins_${DATASET}_semreasoning" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_${DATASET}_semreasoning.parquet" \
    --size 2000 --k 4
cd evaluation
torchrun --nproc_per_node=4 --master_port=5142 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 2000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_${DATASET}_semreasoning.parquet"
py merge.py --model_path "/dev/shm/sft_models/qwen7bins_${DATASET}_semreasoning" --base_model "Qwen/Qwen2.5-7B-Instruct"
rm -rf "/dev/shm/sft_models/qwen7bins_${DATASET}_semreasoning"
source run_eval.sh "ishikauniphore/student_qwen7bins_nemotron_stem_semreasoning"
notify "semreasoning evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis


# HLRep
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "${HF_USERNAME}/generator_qwen7bins_${DATASET}_hlrep" \
    --answer_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_${DATASET}_hlrep.parquet" \
    --size 2000 --k 4
cd evaluation
torchrun --nproc_per_node=4 --master_port=5142 sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" --num_data 2000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/qwen7bins_${DATASET}_hlrep.parquet"
py merge.py --model_path "/dev/shm/sft_models/qwen7bins_${DATASET}_hlrep" --base_model "Qwen/Qwen2.5-7B-Instruct"
rm -rf "/dev/shm/sft_models/qwen7bins_${DATASET}_hlrep"
source run_eval.sh "ishikauniphore/student_qwen7bins_nemotron_stem_hlrep"
notify "hlrep evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*
cd /home/ubuntu/AcquisitionSynthesis


cd /home/ubuntu/AcquisitionSynthesis

notify "now just keeping it alive"
source keep_alive.sh