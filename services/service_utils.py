import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import threading
import queue
import uuid
import torch
from pydantic import BaseModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class RewardsResponse(BaseModel):
    acquisition_reward: float


class WorkerQueue:
    """Serializes GPU calls through a single background worker thread, with dynamic batching."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._results: dict = {}
        self._events: dict = {}
        self._lock = threading.Lock()

    def start(self, fn, *args):
        t = threading.Thread(target=self._worker, args=(fn, args), daemon=True)
        t.start()

    def _worker(self, fn, args):
        while True:
            # Block until at least one item is ready
            job_id, req = self._queue.get()
            batch = [(job_id, req)]
            # Drain any additional pending items without blocking
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            try:
                results = fn([req for _, req in batch], *args)
            except Exception:
                results = [RewardsResponse(acquisition_reward=-5.0)] * len(batch)

            with self._lock:
                for (jid, _), result in zip(batch, results):
                    self._results[jid] = result
                    self._events[jid].set()

    def submit(self, req) -> RewardsResponse:
        job_id = str(uuid.uuid4())
        event = threading.Event()
        with self._lock:
            self._events[job_id] = event
        self._queue.put((job_id, req))
        event.wait()
        with self._lock:
            result = self._results.pop(job_id)
            del self._events[job_id]
        return result


# --- module-level queues (one per service) ---
_confidence_queue: WorkerQueue | None = None
_answerdiff_queue: WorkerQueue | None = None
_semreasoning_queue: WorkerQueue | None = None
_hlrep_queue: WorkerQueue | None = None
_combined_queue: WorkerQueue | None = None
_combined_lai_queue: WorkerQueue | None = None
_combined_lsi_queue: WorkerQueue | None = None


# --- pure compute functions ---
def calculate_answer_confidence(outputs):
    confidence = []
    for output in outputs:
        logprobs = output.outputs[0].logprobs
        ind = 0
        for i in range(len(logprobs)):
            if "answer" in logprobs[i][list(logprobs[i].keys())[0]].decoded_token:
                ind = i
                break
        top1, top2 = [], []
        for lp in logprobs[ind:]:
            keys = list(lp.keys())
            if len(keys) >= 2:
                top1.append(lp[keys[0]].logprob)
                top2.append(lp[keys[1]].logprob)
        if not top1:
            confidence.append(-5.0)
            continue
        avg_diff = (torch.exp(torch.tensor(top1)) - torch.exp(torch.tensor(top2))).mean()
        confidence.append(float(avg_diff))
    return confidence

def calculate_confidence(outputs):
    confidence = []
    for output in outputs:
        logprobs = output.outputs[0].logprobs
        top1, top2 = [], []
        for lp in logprobs:
            keys = list(lp.keys())
            if len(keys) >= 2:
                top1.append(lp[keys[0]].logprob)
                top2.append(lp[keys[1]].logprob)
        if not top1:
            confidence.append(-5.0)
            continue
        avg_diff = (
            torch.exp(torch.tensor(top1)) 
            - 
            torch.exp(torch.tensor(top2))
        ).mean()
        confidence.append(float(avg_diff))
    return confidence

def inference(reqs, language_model, sampling_params):
    questions = [req.data['question'] for req in reqs]
    answers = [req.data['answer'] for req in reqs]
    english_only_instruction = lambda question: (
        "Answer the following question. Reason step-by-step in English inside <reasoning> tags, then output your final answer inside <answer> tags.\n"
        f"<question>\n{question}\n</question>\n"
        "<reasoning>\n\n</reasoning>\n"
        "<answer>\n\n</answer>"
    )
    mcot_instruction = lambda question: (
        "Answer the following question. Reason step-by-step in the language of the question inside <reasoning> tags, then output your final answer inside <answer> tags."
        f"<question> {question} </question>\n"
        "<reasoning> </reasoning>\n"
        "<answer> </answer>"
    )

    english_prompts = [english_only_instruction(q) for q in questions]
    english_outputs = language_model.generate(english_prompts, sampling_params=sampling_params)
    english_parsed_outputs = [o.outputs[0].text.strip() for o in english_outputs]
    
    mcot_prompts = [mcot_instruction(q) for q in questions]
    mcot_outputs = language_model.generate(mcot_prompts, sampling_params=sampling_params)
    mcot_parsed_outputs = [o.outputs[0].text.strip() for o in mcot_outputs]

    return questions, answers, english_outputs, english_parsed_outputs, mcot_outputs, mcot_parsed_outputs







def _compute_confidence(reqs, language_model, sampling_params):
    questions, answers, english_outputs, english_parsed_outputs, mcot_outputs, mcot_parsed_outputs = inference(reqs, language_model, sampling_params)

    eng_confidence = calculate_answer_confidence(english_outputs)
    mcot_confidence = calculate_answer_confidence(mcot_outputs)

    rewards = []
    for eng, mcot, answer, eng_conf, mcot_conf in zip(english_parsed_outputs, mcot_parsed_outputs, answers, eng_confidence, mcot_confidence):
        if eng_conf - mcot_conf > 0.3: # confidence of mcot is significantly bad, we want to increase it, regardless of the answer
            rewards.append(1.0)
        elif abs(eng_conf - mcot_conf) <= 0.3: # pretty much the same confidence, it would be good if mcot is incorect while eng is correct:
            rewards.append(0.0)
        elif mcot_conf - eng_conf > 0.3: # mcot confidence is pretty good, regardless of correctness we don't need to touch these samples
            rewards.append(0.5)
    return [RewardsResponse(acquisition_reward=r) for r in rewards]

def _compute_answerdiff(reqs, language_model, sampling_params):
    questions, answers, english_outputs, english_parsed_outputs, mcot_outputs, mcot_parsed_outputs = inference(reqs, language_model, sampling_params)

    rewards = []
    for eng, mcot, answer in zip(english_parsed_outputs, mcot_parsed_outputs, answers):
        eng_correct = answer in eng
        mcot_correct = answer in mcot
        if eng_correct and not mcot_correct: # representation gap
            rewards.append(1.0)
        elif (mcot_correct and not eng_correct) or (not mcot_correct and not eng_correct): # representation gap OR understanding gap
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return [RewardsResponse(acquisition_reward=r) for r in rewards]

def _compute_semreasoning(reqs, language_model, sampling_params, embedding_model):
    questions, answers, english_outputs, english_parsed_outputs, mcot_outputs, mcot_parsed_outputs = inference(reqs, language_model, sampling_params)
    eng_reasoning = [e.split("<reasoning>")[-1].split("</reasoning>")[0] for e in english_parsed_outputs]
    mcot_reasoning = [m.split("<reasoning>")[-1].split("</reasoning>")[0] for m in mcot_parsed_outputs]

    eng_embeddings = embedding_model.embed(eng_reasoning)
    eng_embeddings = torch.Tensor([a.outputs.embedding for a in eng_embeddings])
    mcot_embeddings = embedding_model.embed(mcot_reasoning)
    mcot_embeddings = torch.Tensor([a.outputs.embedding for a in mcot_embeddings])

    results = []
    for eng, mcot in zip(eng_embeddings, mcot_embeddings):
        dist = 1.0 - F.cosine_similarity(eng.unsqueeze(0), mcot.unsqueeze(0)).item()
        results.append(RewardsResponse(acquisition_reward=dist))
    return results

def _compute_hlrep(reqs, tokenizer, model):
    results = []
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

    for req in reqs:
        question = req.data['question']

        eng_inputs = tokenizer(english_only_instruction(question), return_tensors='pt').to(device)
        mcot_inputs = tokenizer(mcot_instruction(question), return_tensors='pt').to(device)
        eng_prompt_len = eng_inputs['input_ids'].shape[1]
        mcot_prompt_len = mcot_inputs['input_ids'].shape[1]

        with torch.no_grad():
            eng_gen_ids = model.generate(
                **eng_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
            mcot_gen_ids = model.generate(
                **mcot_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        eng_new_ids = eng_gen_ids[0][eng_prompt_len:].tolist()
        mcot_new_ids = mcot_gen_ids[0][mcot_prompt_len:].tolist()

        eng_reason_end = find_reasoning_end(eng_new_ids)
        mcot_reason_end = find_reasoning_end(mcot_new_ids)

        if eng_reason_end == 0 or mcot_reason_end == 0:
            results.append(RewardsResponse(acquisition_reward=0.0))
            continue

        with torch.no_grad():
            eng_out = model(input_ids=eng_gen_ids, output_hidden_states=True)
            mcot_out = model(input_ids=mcot_gen_ids, output_hidden_states=True)

        # Last layer hidden states: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
        eng_hidden = eng_out.hidden_states[-1][0]
        mcot_hidden = mcot_out.hidden_states[-1][0]

        # Mean-pool over the reasoning token positions (generated tokens up to </reasoning>)
        eng_rep = eng_hidden[eng_prompt_len:eng_prompt_len + eng_reason_end].mean(dim=0)
        mcot_rep = mcot_hidden[mcot_prompt_len:mcot_prompt_len + mcot_reason_end].mean(dim=0)

        cos_dist = 1.0 - F.cosine_similarity(eng_rep.unsqueeze(0), mcot_rep.unsqueeze(0)).item()
        results.append(RewardsResponse(acquisition_reward=cos_dist))

    return results






def calculate_answer_uncertainty_from_scores(scores, tokenizer):
    """Same top1/top2-logprob-gap confidence as calculate_answer_confidence, but
    walking HF generate()'s per-step `scores` (from output_scores=True) instead of
    vLLM's per-position logprobs dicts."""
    ind = 0
    for i, step_logits in enumerate(scores):
        top1_id = int(step_logits[0].argmax())
        if "answer" in tokenizer.decode([top1_id]):
            ind = i
            break

    top1, top2 = [], []
    for step_logits in scores[ind:]:
        logprobs = F.log_softmax(step_logits[0], dim=-1)
        vals, _ = logprobs.topk(2)
        top1.append(vals[0].item())
        # top2.append(vals[1].item())

    if not top1:
        return -5.0
    diffs = torch.exp(torch.tensor(top1))
    avg_diff = (torch.ones_like(diffs) - diffs).mean() #(1.0/(torch.exp(torch.tensor(top1)) - torch.exp(torch.tensor(top2)))).mean()
    return float(avg_diff)

def _compute_combined(reqs, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    results = []
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

    for req in reqs:
        question = req.data['question']

        eng_inputs = tokenizer(english_only_instruction(question), return_tensors='pt').to(device)
        mcot_inputs = tokenizer(mcot_instruction(question), return_tensors='pt').to(device)
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
        eng_confidence = calculate_answer_uncertainty_from_scores(eng_gen_out.scores, tokenizer)
        conf = min(eng_confidence, 1.0)

        eng_new_ids = eng_gen_ids[0][eng_prompt_len:].tolist()
        mcot_new_ids = mcot_gen_ids[0][mcot_prompt_len:].tolist()

        eng_reason_end = find_reasoning_end(eng_new_ids)
        mcot_reason_end = find_reasoning_end(mcot_new_ids)

        if eng_reason_end == 0 or mcot_reason_end == 0:
            results.append(RewardsResponse(acquisition_reward=0.0))
            continue

        with torch.no_grad():
            eng_out = model(input_ids=eng_gen_ids, output_hidden_states=True)
            mcot_out = model(input_ids=mcot_gen_ids, output_hidden_states=True)

        # Last layer hidden states: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
        eng_hidden = eng_out.hidden_states[-1][0]
        mcot_hidden = mcot_out.hidden_states[-1][0]

        # Mean-pool over the reasoning token positions (generated tokens up to </reasoning>)
        eng_rep = eng_hidden[eng_prompt_len:eng_prompt_len + eng_reason_end].mean(dim=0)
        mcot_rep = mcot_hidden[mcot_prompt_len:mcot_prompt_len + mcot_reason_end].mean(dim=0)

        cos_dist = 1.0 - F.cosine_similarity(eng_rep.unsqueeze(0), mcot_rep.unsqueeze(0)).item()
        rep = cos_dist
        results.append(RewardsResponse(acquisition_reward=conf + rep))

    return results


def _compute_combined_lai(reqs, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    """Ablation: LAI (confidence) term only. Identical generation/scoring pipeline
    to _compute_combined, but the reward is just `conf` instead of `conf + rep`."""
    results = []
    device = next(model.parameters()).device

    english_only_instruction = lambda question: (
        "Answer the following question. Reason step-by-step in English inside <reasoning> tags, then output your final answer inside <answer> tags.\n"
        f"<question>\n{question}\n</question>\n"
        "<reasoning>"
    )

    for req in reqs:
        question = req.data['question']

        eng_inputs = tokenizer(english_only_instruction(question), return_tensors='pt').to(device)

        with torch.no_grad():
            eng_gen_out = model.generate(
                **eng_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True, return_dict_in_generate=True,
            )

        eng_confidence = calculate_answer_uncertainty_from_scores(eng_gen_out.scores, tokenizer)
        conf = min(eng_confidence, 1.0)
        results.append(RewardsResponse(acquisition_reward=conf))

    return results


def _compute_combined_lsi(reqs, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    """Ablation: LSI (hidden-state representation gap) term only. Identical
    generation/scoring pipeline to _compute_combined, but the reward is just
    `rep` instead of `conf + rep`."""
    results = []
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

    for req in reqs:
        question = req.data['question']

        eng_inputs = tokenizer(english_only_instruction(question), return_tensors='pt').to(device)
        mcot_inputs = tokenizer(mcot_instruction(question), return_tensors='pt').to(device)
        eng_prompt_len = eng_inputs['input_ids'].shape[1]
        mcot_prompt_len = mcot_inputs['input_ids'].shape[1]

        with torch.no_grad():
            eng_gen_ids = model.generate(
                **eng_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
            mcot_gen_ids = model.generate(
                **mcot_inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        eng_new_ids = eng_gen_ids[0][eng_prompt_len:].tolist()
        mcot_new_ids = mcot_gen_ids[0][mcot_prompt_len:].tolist()

        eng_reason_end = find_reasoning_end(eng_new_ids)
        mcot_reason_end = find_reasoning_end(mcot_new_ids)

        if eng_reason_end == 0 or mcot_reason_end == 0:
            results.append(RewardsResponse(acquisition_reward=0.0))
            continue

        with torch.no_grad():
            eng_out = model(input_ids=eng_gen_ids, output_hidden_states=True)
            mcot_out = model(input_ids=mcot_gen_ids, output_hidden_states=True)

        eng_hidden = eng_out.hidden_states[-1][0]
        mcot_hidden = mcot_out.hidden_states[-1][0]

        eng_rep = eng_hidden[eng_prompt_len:eng_prompt_len + eng_reason_end].mean(dim=0)
        mcot_rep = mcot_hidden[mcot_prompt_len:mcot_prompt_len + mcot_reason_end].mean(dim=0)

        cos_dist = 1.0 - F.cosine_similarity(eng_rep.unsqueeze(0), mcot_rep.unsqueeze(0)).item()
        results.append(RewardsResponse(acquisition_reward=cos_dist))

    return results


# --- init functions (called from all.py on service start) ---

def init_confidence_worker(language_model, sampling_params):
    global _confidence_queue
    _confidence_queue = WorkerQueue()
    _confidence_queue.start(_compute_confidence, language_model, sampling_params)

def init_answerdiff_worker(language_model, sampling_params):
    global _answerdiff_queue
    _answerdiff_queue = WorkerQueue()
    _answerdiff_queue.start(_compute_answerdiff, language_model, sampling_params)

def init_semreasoning_worker(language_model, sampling_params, embedding_model):
    global _semreasoning_queue
    _semreasoning_queue = WorkerQueue()
    _semreasoning_queue.start(_compute_semreasoning, language_model, sampling_params, embedding_model)

def init_hlrep_worker(tokenizer, model):
    global _hlrep_queue
    _hlrep_queue = WorkerQueue()
    _hlrep_queue.start(_compute_hlrep, tokenizer, model)

def init_combined_worker(language_tokenizer, language_auto_model):
    global _combined_queue
    _combined_queue = WorkerQueue()
    _combined_queue.start(_compute_combined, language_tokenizer, language_auto_model)

def init_combined_lai_worker(language_tokenizer, language_auto_model):
    global _combined_lai_queue
    _combined_lai_queue = WorkerQueue()
    _combined_lai_queue.start(_compute_combined_lai, language_tokenizer, language_auto_model)

def init_combined_lsi_worker(language_tokenizer, language_auto_model):
    global _combined_lsi_queue
    _combined_lsi_queue = WorkerQueue()
    _combined_lsi_queue.start(_compute_combined_lsi, language_tokenizer, language_auto_model)


# --- public API (called from all.py routes) ---

def confidence(req):
    if _confidence_queue is None:
        raise RuntimeError("confidence worker not initialized — call /start_service first")
    return _confidence_queue.submit(req)

def answerdiff(req):
    if _answerdiff_queue is None:
        raise RuntimeError("answerdiff worker not initialized — call /start_service first")
    return _answerdiff_queue.submit(req)

def semreasoning(req):
    if _semreasoning_queue is None:
        raise RuntimeError("semreasoning worker not initialized — call /start_service first")
    return _semreasoning_queue.submit(req)

def hlrep(req):
    if _hlrep_queue is None:
        raise RuntimeError("hlrep worker not initialized — call /start_service first")
    return _hlrep_queue.submit(req)

def combined(req):
    if _combined_queue is None:
        raise RuntimeError("combined worker not initialized — call /start_service first")
    return _combined_queue.submit(req)

def combined_lai(req):
    if _combined_lai_queue is None:
        raise RuntimeError("combined_lai worker not initialized — call /start_service first")
    return _combined_lai_queue.submit(req)

def combined_lsi(req):
    if _combined_lsi_queue is None:
        raise RuntimeError("combined_lsi worker not initialized — call /start_service first")
    return _combined_lsi_queue.submit(req)