import os
import gc
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from argparse import ArgumentParser
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from glob import glob

n = 500

def perform_nemotron_stem_inference(llm, sampling_params, experiment):
    prompt_template = lambda q: f"Answer the following multiple choice question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer (the letter of the answer choice) in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in experiment['questions']]
    outputs = []

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]
    
    return [{
        "experiment_name": "nemotron_stem",
        "task": "classification",
        "questions": experiment['questions'],
        "answers": experiment['answers'],
        "outputs": outputs
    }]

def perform_nemotron_math_inference(llm, sampling_params, experiment):
    prompt_template = lambda q: f"Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in experiment['questions']]
    outputs = []

    outputs = llm.generate(prompts[:n], sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]
    
    return [{
        "experiment_name": "nemotron_math",
        "task": "math",
        "questions": experiment['questions'],
        "answers": experiment['answers'],
        "outputs": outputs
    }]

def perform_nemotron_chat_inference(llm, sampling_params, experiment):
    prompt_template = lambda q: f"Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in experiment['questions']]
    outputs = []

    outputs = llm.generate(prompts[:n], sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]
    
    return [{
        "experiment_name": "nemotron_chat",
        "task": "open-ended",
        "questions": experiment['questions'],
        "answers": experiment['answers'],
        "outputs": outputs
    }]

def perform_mhotpot_inference(llm, sampling_params, experiment):
    queries = experiment['query']
    contexts = experiment['context']

    prompt_template = lambda c, q: f"Given some context, the task is the answer the question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> </answer> tags.\n\n<context> {c} </context>\n<question> {q} </question>."
    prompts = [prompt_template(c, q) for c, q in zip(contexts, queries)]
    outputs = []

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]

    experiment['outputs'] = outputs

    return experiment


def perform_mmmlu_inference(llm, sampling_params, experiment):
    questions = experiment['questions']

    prompt_template = lambda q: f"Answer the following multiple choice question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer (the letter of the answer choice) in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in questions]
    outputs = []

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]

    experiment['outputs'] = outputs

    return experiment

def perform_opus_inference(llm, sampling_params, experiment):
    questions = experiment['questions']

    prompt_template = lambda q: f"Answer the question and output your final answer in <answer> </answer> tags.\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in questions]
    outputs = []

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]

    experiment['outputs'] = outputs

    return experiment
