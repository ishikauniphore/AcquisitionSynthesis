DATASET="nemotron"

### STEP 2: Formatting
export CUDA_VISIBLE_DEVICES=0,1
py generating_data/baseline_selection_filtered.py --data "nemotron" --model "Qwen/Qwen2.5-7B-Instruct" --dir_name "qwen7b_nemotron" --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/filtered_nemotron_qwen7bins.parquet" --size 5000
notify "[baselines] filter done on qwen7b"
source keep_alive.sh

export CUDA_VISIBLE_DEVICES=2,3
py generating_data/baseline_selection_filtered.py --data "nemotron" --model "meta-llama/Llama-3.1-8B-Instruct" --dir_name "llama8b_nemotron" --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/filtered_nemotron_llama8bins.parquet" --size 5000
notify "[baselines] filter done on llama8b"
source keep_alive.sh

export CUDA_VISIBLE_DEVICES=2,3
py generating_data/baseline_selection_filtered.py --data "nemotron" --model "Qwen/Qwen2.5-14B-Instruct" --dir_name "qwen14b_nemotron" --output_file "/home/ubuntu/AcquisitionSynthesis/training_data/filtered_nemotron_qwen14bins.parquet" --size 5000
notify "[baselines] filter done on qwen14b"
source keep_alive.sh


### STEP 3: Student evaluation
cd evaluation
rm -rf /dev/shm/sft_models/
torchrun --nproc_per_node=2 sft.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" --num_data 5000 \
    --file_name "/home/ubuntu/AcquisitionSynthesis/training_data/filtered_nemotron_llama8bins.parquet"
py merge.py --model_path "/dev/shm/sft_models/filtered_nemotron_llama8bins" --base_model "meta-llama/Llama-3.1-8B-Instruct"
source run_eval.sh "ishikauniphore/student_filtered_nemotron_llama8bins"

cd ..
notify "filtered done for llama"

source keep_alive.sh


