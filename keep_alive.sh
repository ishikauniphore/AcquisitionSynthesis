export CUDA_VISIBLE_DEVICES=4,5,6,7
cd evaluation
while :
do
    source run_eval.sh "Qwen/Qwen2.5-7B-Instruct"
done