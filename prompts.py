def get_prompt_template(data_name, multilingual=False, english_reasoning=False):
    """Prompt templates shared by SFT training (evaluation/sft.py), synthetic
    answer generation (generating_data/data_gen_cluster.py), and eval
    (evaluation/model_inference_utils.py). Keep this the single source so the
    call sites can't drift out of sync with each other again.
    """
    if english_reasoning:
        reasoning_clause = "Output your reasoning in ENGLISH in the <reasoning> </reasoning> tags"
    else:
        reasoning_clause = "Output your reasoning in <reasoning> </reasoning> tags"
    lang_clause = " Answer in the same language as the question." if multilingual else ""
    if "stem" in data_name:
        return lambda q: f"Answer the following multiple choice question.{lang_clause} {reasoning_clause}, and your final answer (the letter of the answer choice) in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    elif "math" in data_name:
        return lambda q: f"Answer the following question.{lang_clause} {reasoning_clause}, and your final answer in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    elif "chat" in data_name:
        return lambda q: f"Answer the following question.{lang_clause} {reasoning_clause}, and your final answer in <answer> </answer> tags.\n\n<question> {q} </question>."
    elif "nemotron" in data_name:
        # nemotron mixes stem/math/chat questions under one data_name, so the
        # MCQ instruction has to be unconditional (it's a no-op for non-MCQ
        # questions) since we can't tell the sub-type per example.
        mcq_clause = " If the question is multiple choice, put only the letter of the correct answer choice in the box."
        return lambda q: f"Answer the following question.{lang_clause} {reasoning_clause}, and your final answer in <answer> \\boxed{{}} </answer> tags.{mcq_clause}\n\n<question> {q} </question>."
    else:
        raise ValueError(f"Unknown data name: {data_name}")