from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
from tqdm import tqdm

def calculate_answer_uncertainty_from_scores(scores, tokenizer, batch_idx=0):
    """Same top1/top2-logprob-gap confidence as calculate_answer_confidence, but
    walking HF generate()'s per-step `scores` (from output_scores=True) instead of
    vLLM's per-position logprobs dicts. `scores` is the full batch's per-step
    logits; `batch_idx` selects the row within it."""
    ind = 0
    for i, step_logits in enumerate(scores):
        top1_id = int(step_logits[batch_idx].argmax())
        if "answer" in tokenizer.decode([top1_id]):
            ind = i
            break

    top1, top2 = [], []
    for step_logits in scores[ind:]:
        logprobs = F.log_softmax(step_logits[batch_idx], dim=-1)
        vals, _ = logprobs.topk(2)
        top1.append(vals[0].item())
        # top2.append(vals[1].item())

    if not top1:
        return -5.0
    diffs = torch.exp(torch.tensor(top1))
    avg_diff = (torch.ones_like(diffs) - diffs).mean() #(1.0/(torch.exp(torch.tensor(top1)) - torch.exp(torch.tensor(top2)))).mean()
    return float(avg_diff)

def _compute_combined(reqs, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    device = next(model.parameters()).device

    english_only_instruction = lambda question: (
        "Answer the following question. Reason step-by-step in English inside <reasoning> tags, then output your final answer inside <answer> tags.\n"
        f"<question>\n{question}\n</question>\n"
        "<reasoning>"
    )
    mcot_instruction = lambda question: (
        "Answer the following question. Reason step-by-step in the language of the question inside <reasoning> tags, then output your final answer inside <answer> tags.\n"
        f"<question>{question}</question>\n"
        "<reasoning>"
    )

    reasoning_close_ids = tokenizer("</reasoning>", add_special_tokens=False).input_ids

    def find_reasoning_end(ids_list):
        for i in range(len(ids_list) - len(reasoning_close_ids) + 1):
            if ids_list[i:i + len(reasoning_close_ids)] == reasoning_close_ids:
                return i
        return len(ids_list)

    results = [0.0] * len(reqs)

    eng_prompts = [english_only_instruction(q) for q in reqs]
    mcot_prompts = [mcot_instruction(q) for q in reqs]

    eng_inputs = tokenizer(eng_prompts, return_tensors='pt', padding=True).to(device)
    mcot_inputs = tokenizer(mcot_prompts, return_tensors='pt', padding=True).to(device)
    # Left padding means every row's prompt ends at the same index, so newly
    # generated tokens start at this offset for the whole batch.
    eng_prompt_len = eng_inputs['input_ids'].shape[1]
    mcot_prompt_len = mcot_inputs['input_ids'].shape[1]

    with torch.no_grad():
        eng_gen_out = model.generate(
            **eng_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
        mcot_gen_ids = model.generate(
            **mcot_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    eng_gen_ids = eng_gen_out.sequences

    confs = []
    for b in range(len(reqs)):
        eng_confidence = calculate_answer_uncertainty_from_scores(eng_gen_out.scores, tokenizer, batch_idx=b)
        confs.append(min(eng_confidence, 1.0))

    eng_new_ids = eng_gen_ids[:, eng_prompt_len:].tolist()
    mcot_new_ids = mcot_gen_ids[:, mcot_prompt_len:].tolist()

    eng_reason_ends = [find_reasoning_end(ids) for ids in eng_new_ids]
    mcot_reason_ends = [find_reasoning_end(ids) for ids in mcot_new_ids]

    valid = [b for b in range(len(reqs)) if eng_reason_ends[b] != 0 and mcot_reason_ends[b] != 0]

    if valid:
        eng_valid_ids = eng_gen_ids[valid]
        mcot_valid_ids = mcot_gen_ids[valid]
        # pad_token_id == eos_token_id here, so this also strips left-padding
        # from the prompt as well as trailing padding after generation ends.
        eng_attn = (eng_valid_ids != tokenizer.pad_token_id).long()
        mcot_attn = (mcot_valid_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            eng_out = model(input_ids=eng_valid_ids, attention_mask=eng_attn, output_hidden_states=True)
            mcot_out = model(input_ids=mcot_valid_ids, attention_mask=mcot_attn, output_hidden_states=True)

        # Last layer hidden states: (valid_batch, seq_len, hidden_dim)
        eng_hidden = eng_out.hidden_states[-1]
        mcot_hidden = mcot_out.hidden_states[-1]

        for vi, b in enumerate(valid):
            # Mean-pool over the reasoning token positions (generated tokens up to </reasoning>)
            eng_rep = eng_hidden[vi, eng_prompt_len:eng_prompt_len + eng_reason_ends[b]].mean(dim=0)
            mcot_rep = mcot_hidden[vi, mcot_prompt_len:mcot_prompt_len + mcot_reason_ends[b]].mean(dim=0)

            cos_dist = 1.0 - F.cosine_similarity(eng_rep.unsqueeze(0), mcot_rep.unsqueeze(0)).item()
            results[b] = confs[b] + cos_dist

    return results

def compute_combined_batch(reqs, mname, batch_size=4):
    rewards = []
    auto_model = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(mname)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    for i in tqdm(range(0, len(reqs), batch_size)):
        rewards.extend(_compute_combined(reqs[i:i+batch_size], tokenizer, auto_model))
    return rewards