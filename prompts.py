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
    else:
        raise ValueError(f"Unknown data name: {data_name}")


def canonicalize_answer(answer_text, data_name):
    """Reduce a raw <answer> block to the same canonical form eval uses when
    scoring model predictions (see the boxed/letter extraction in
    evaluation/model_inference_utils.py), so labels produced at generation
    time and predictions scored at eval time are directly comparable.
    """
    text = answer_text.strip()
    if "\\boxed{" in text:
        text = text.split("\\boxed{")[-1].split("}")[0].strip()
    if "stem" in data_name and len(text) > 1 and text[0].isalpha() and text[1] in ":).":
        text = text[0].upper()
    return text
