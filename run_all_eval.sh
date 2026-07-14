export CUDA_VISIBLE_DEVICES=0,1,2,3
### English Reasoning
# source run_englishreasoning.sh
# notify "english reasoning evals done!!!!"
# rm -rf /dev/shm/hf_cache/hub/*

cd evaluation

### Base Models
source run_eval.sh "Qwen/Qwen2.5-7B-Instruct"
# source run_eval.sh "Qwen/Qwen2.5-14B-Instruct"
# source run_eval.sh "meta-llama/Llama-3.1-8B-Instruct"
notify "base model evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*

### Selection
source run_eval.sh "ishikauniphore/student_selected_nemotron_stem"
notify "selection evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*

### DataEnvGym
# source run_eval.sh "ishikauniphore/student_DataEnvGym_nemotron_stem_qwen7bins"
# source run_eval.sh "ishikauniphore/student_DataEnvGym_nemotron_stem_qwen14bins"
# source run_eval.sh "ishikauniphore/student_DataEnvGym_nemotron_stem_llama8bins"
notify "dataenvgym evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*

### Untrained Data Generator
source run_eval.sh "ishikauniphore/student_Original_nemotron_stem_qwen7bins"
# source run_eval.sh "ishikauniphore/student_Original_nemotron_stem_qwen14bins"
# source run_eval.sh "ishikauniphore/student_Original_nemotron_stem_llama8bins"
notify "untrained data generator evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*

### TANDEM
# AnswerDiff
source run_eval.sh "ishikauniphore/student_qwen7bins_nemotron_stem_answerdiff"
# source run_eval.sh "ishikauniphore/student_qwen14bins_nemotron_stem_answerdiff"
# source run_eval.sh "ishikauniphore/student_llama8bins_nemotron_stem_answerdiff"
notify "answerdiff evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*

# Confidence
source run_eval.sh "ishikauniphore/student_qwen7bins_nemotron_stem_confidence"
# source run_eval.sh "ishikauniphore/student_qwen14bins_nemotron_stem_confidence"
# source run_eval.sh "ishikauniphore/student_llama8bins_nemotron_stem_confidence"
notify "confidence evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*

# SemanticReasoning
source run_eval.sh "ishikauniphore/student_qwen7bins_nemotron_stem_semreasoning"
# source run_eval.sh "ishikauniphore/student_qwen14bins_nemotron_stem_semreasoning"
# source run_eval.sh "ishikauniphore/student_llama8bins_nemotron_stem_semreasoning"
notify "semantic reasoning evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*

# HLRep
source run_eval.sh "ishikauniphore/student_qwen7bins_nemotron_stem_hlrep"
# source run_eval.sh "ishikauniphore/student_qwen14bins_nemotron_stem_hlrep"
# source run_eval.sh "ishikauniphore/student_llama8bins_nemotron_stem_hlrep"
notify "hlrep evals done!!!!"
rm -rf /dev/shm/hf_cache/hub/*

cd /home/ubuntu/AcquisitionSynthesis

notify "now just keeping it alive"
source keep_alive.sh