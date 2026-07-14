export CUDA_VISIBLE_DEVICES=0,1,2,3
cd evaluation
while :
do
    source run_eval.sh "Qwen/Qwen2.5-7B-Instruct"
done