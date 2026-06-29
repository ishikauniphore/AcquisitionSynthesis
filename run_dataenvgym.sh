# cd generating_data
# py baseline_synthesis_DataEnvGym.py \
#     --data "nemotron_stem" \
#     --model "Qwen/Qwen2.5-7B-Instruct" \
#     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_nemotron_stem_qwen7bins.parquet"
# py baseline_synthesis_DataEnvGym.py \
#     --data "nemotron_stem" \
#     --model "Qwen/Qwen2.5-14B-Instruct" \
#     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_nemotron_stem_qwen14bins.parquet"
# py baseline_synthesis_DataEnvGym.py \
#     --data "nemotron_stem" \
#     --model "meta-llama/Llama-3.1-8B-Instruct" \
#     --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_nemotron_stem_llama8bins.parquet"
# cd ..



cd /home/ubuntu/AcquisitionSynthesis/evaluation
# torchrun --nproc_per_node=4 sft.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_nemotron_stem_qwen7bins.parquet"
# py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_nemotron_stem_qwen7bins"
# rm -rf /dev/shm/sft_models/DataEnvGym_nemotron_stem_qwen7bins

torchrun --nproc_per_node=4 sft.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_nemotron_stem_qwen14bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_nemotron_stem_qwen14bins" --base_model "Qwen/Qwen2.5-14B-Instruct"
rm -rf /dev/shm/sft_models/DataEnvGym_nemotron_stem_qwen14bins

torchrun --nproc_per_node=4 sft.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/DataEnvGym_nemotron_stem_llama8bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/DataEnvGym_nemotron_stem_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"
rm -rf /dev/shm/sft_models/DataEnvGym_nemotron_stem_llama8bins



# source run_eval.sh "ishikauniphore/student_DataEnvGym_nemotron_stem_qwen7bins"
source run_eval.sh "ishikauniphore/student_DataEnvGym_nemotron_stem_qwen14bins"
source run_eval.sh "ishikauniphore/student_DataEnvGym_nemotron_stem_llama8bins"
cd ..

notify "dataenv gym done"


# cd evaluation
# source run_eval.sh "Qwen/Qwen2.5-14B-Instruct"
# source run_eval.sh "meta-llama/Llama-3.1-8B-Instruct"