import os
import sys
import gc
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import csv
import torch
from argparse import ArgumentParser
from config import *
import evaluate
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import canonicalize_answer

def evaluate_student(predictions, references):
    rouge_metric = evaluate.load('rouge')

    scores = []
    for p, r in zip(predictions, references):
        if p is None:
            p = " "
        if r is None: 
            r = " "
        scores.append(rouge_metric.compute(predictions=[p], references=[r])['rouge1'])
    del rouge_metric
    return np.array(scores).reshape((-1,))

def format_conversation(row):
    return f"## User: {row['prompt']}\n## Assistant: {row['completion']}"

def evaluation(model, dataset):
    if not os.path.exists(CACHE_FILE):
        student_model = LLM(model, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.8, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
        prompts = [f"## User: {p}\n## Assistant: " for p in list(dataset['question'])]
        languages = list(dataset['language'])
        outputs = student_model.generate(prompts, sampling_params=sampling_params)
        outputs = [o.outputs[0].text.strip().replace("\n", " ") for o in outputs]
        prompts = [p.strip().replace("\n", " ") for p in prompts]

        pd.DataFrame.from_dict({
            "prompt": prompts,
            "completion": outputs,
            "language": languages
        }).to_csv(CACHE_FILE, sep="|", quoting=csv.QUOTE_ALL)

        destroy_model_parallel()
        destroy_distributed_environment()
        del student_model, sampling_params
        gc.collect()
        torch.cuda.empty_cache()
        for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE']:
            os.environ.pop(key, None)

def data_gen_policy(dataset, threshold=0.5):
    df = pd.read_csv(CACHE_FILE, sep="|", quoting=csv.QUOTE_ALL)
    
    if "similarity_score" not in list(df.columns):
        similarity_score = evaluate_student(list(df['completion']), list(dataset['answer']))
        df['similarity_score'] = similarity_score
        df.to_csv(CACHE_FILE, sep="|", quoting=csv.QUOTE_ALL)

    threshold = float(df['similarity_score'].mean())
    return df[df['similarity_score'] <= threshold]

def parse_html(generated_samples, dataset_name):
    synthetic_data = []
    for text in generated_samples:
        try:
            sample_pattern = re.compile(
                r'<question>(.*?)</question>\s*'
                r'<reasoning>(.*?)</reasoning>\s*'
                r'<answer>(.*?)</answer>\s*',
                re.DOTALL | re.IGNORECASE
            )

            match = sample_pattern.search(text)
            if not match:
                continue

            synthetic_data.append({
                "question": match.group(1).strip(),
                "reasoning": match.group(2).strip(),
                "answer": canonicalize_answer(match.group(3).strip(), dataset_name)
            })
        except:
            continue

    return synthetic_data

def data_gen_engine(mistakes: pd.DataFrame, teacher_name, num_samples, dataset_name):
    teacher_model = LLM(teacher_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.8, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=300)

    # Match the answer format the student is actually trained/scored on
    # (evaluation/sft.py, prompts.py) - otherwise the teacher answers however
    # it likes and the label format won't line up with the SFT prompt template.
    if "stem" in dataset_name:
        answer_instruction = lambda lang: (
            "The generated question must be multiple choice with lettered options (A, B, C, ...). "
            "Output ONLY the letter of the correct choice here - no words, no punctuation."
        )
    elif "math" in dataset_name:
        answer_instruction = lambda lang: "Output ONLY the final answer here as \\boxed{} - no explanation."
    else:
        answer_instruction = lambda lang: f"Output the final answer in {lang} plain text."

    prompt = lambda row: f"""You are a experienced machine learning engineer and your role is create training data for a model. We will focus on improving skills related to the following mistake made by a student:
{format_conversation(row)}

You have two tasks. First, identify the particular skill that the student is weak in. Second, generate a training sample IN {row['language']} that targets the skill for the student to learn. Format your answer in the below tags:

<weak_skill>
Output the particular skill that the student is weak in, given the conversation.
</weak_skill>

<question>
Output the question in {row['language']} of the generated target sample that will help the student improve.
</question>

<reasoning>
Output the reasoning in {row['language']} behind the solution to the generated question.
</reasoning>

<answer>
{answer_instruction(row['language'])}
</answer>
"""

    generated_dataset = []
    inputs = [prompt(row) for _, row in mistakes.iterrows()]

    while len(generated_dataset) < num_samples:
        outputs = teacher_model.generate(inputs, sampling_params=sampling_params)
        outputs = [o.outputs[0].text.strip() for o in outputs]

        generated_dataset.extend(parse_html(outputs, dataset_name))
        print(f"got {len(generated_dataset)}/{num_samples}...")

        pd.DataFrame.from_records(generated_dataset).to_parquet(GEN_SAMPLES_FILE)
    
    pd.DataFrame.from_records(generated_dataset).to_parquet(GEN_SAMPLES_FILE)

    destroy_model_parallel()
    destroy_distributed_environment()
    del teacher_model, sampling_params
    gc.collect()
    torch.cuda.empty_cache()
    for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE']:
        os.environ.pop(key, None)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--data", type=str, default="numina")
    argparser.add_argument("--student_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    argparser.add_argument("--teacher_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    argparser.add_argument("--output_file", type=str, default="dataenv_dataset.parquet")
    argparser.add_argument("--size", type=int, default=1000)
    args = argparser.parse_args()

    dataset = load_data(f"/home/ubuntu/AcquisitionSynthesis/data/{args.data}/valid.parquet")
    # dataset = dataset[:100]
    CACHE_FILE = 'dataenvgym_cache.csv'

    # the steps are derived from Figure 1 in https://arxiv.org/pdf/2410.06215

    # Step 1: Evaluation
    evaluation(args.student_model, dataset)

    # Step 2: Data Generation Policy
    mistakes = data_gen_policy(dataset)
    print('\n\n\nNumber of mistakes made: ', len(mistakes), '\n\n\n')

    # Step 3: Data Generation Engine
    GEN_SAMPLES_FILE = args.output_file
    data_gen_engine(mistakes, teacher_name=args.teacher_model, num_samples=args.size, dataset_name=args.data)
    os.remove(CACHE_FILE)