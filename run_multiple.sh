export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_NAMES=("ishikauniphore/student_SelectedGEN_nemotron_qwen7bins" "ishikauniphore/student_SelectedGT_qwen7bins_nemotron" "ishikauniphore/student_Original_nemotron_qwen7bins" "ishikauniphore/student_DataEnvGym_nemotron_qwen7bins" "ishikauniphore/student_SelectedGEN_nemotron_llama8bins" "ishikauniphore/student_SelectedGT_llama8bins_nemotron" "ishikauniphore/student_DataEnvGym_nemotron_llama8bins" "ishikauniphore/student_qwen14bins_nemotron_combined" "ishikauniphore/student_Original_nemotron_llama8bins" "ishikauniphore/student_qwen7bins_nemotron_combined" "ishikauniphore/student_llama8bins_nemotron_combined" "ishikauniphore/student_SelectedGEN_nemotron_qwen14bins" "ishikauniphore/student_SelectedGT_qwen14bins_nemotron" "ishikauniphore/student_Original_nemotron_qwen14bins" "ishikauniphore/student_DataEnvGym_nemotron_qwen14bins")

# "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "meta-llama/Llama-3.1-8B-Instruct"

cd evaluation
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    source run_eval.sh "${MODEL_NAME}"
    rm -rf /dev/shm/hf_cache/hub/*ishika*
done
notify "round 2 of multiple done"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    source run_eval.sh "${MODEL_NAME}"
    rm -rf /dev/shm/hf_cache/hub/*ishika*
done
notify "round 3 of multiple done"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    source run_eval.sh "${MODEL_NAME}"
    rm -rf /dev/shm/hf_cache/hub/*ishika*
done
notify "round 4 of multiple done"


cd /home/ubuntu/AcquisitionSynthesis
source keep_alive.sh