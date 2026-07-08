import os
import argparse
from accelerate import PartialState
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch

def get_prompt_template(data_name):
    """Must match the eval prompts in evaluation/model_inference_utils.py exactly,
    so the model sees the same instruction phrasing at train and eval time."""
    if "stem" in data_name:
        return lambda q: f"Answer the following multiple choice question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer (the letter of the answer choice) in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    elif "math" in data_name:
        return lambda q: f"Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    elif "chat" in data_name:
        return lambda q: f"Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> </answer> tags.\n\n<question> {q} </question>."
    else:
        raise ValueError(f"Unknown data name: {data_name}")


def format_prompt(example, tokenizer, prompt_template):
    messages = [
        {"role": "user", "content": prompt_template(example['question'])},
        {"role": "assistant", "content": f"<reasoning> {example['reasoning']} </reasoning>\n<answer> {example['answer']} </answer>"},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def sft_train(file_name, model_name, num_epochs=5, output_dir="/dev/shm/sft_models/", student_name=None, n=1000):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_parquet(file_name)[:n]
    assert "question" in df.columns and "answer" in df.columns and "reasoning" in df.columns, \
        "CSV must have 'question' and 'answer' columns"

    prompt_template = get_prompt_template(file_name)
    dataset = Dataset.from_pandas(df[["question", "answer", "reasoning"]].dropna())
    dataset = dataset.map(lambda ex: format_prompt(ex, tokenizer, prompt_template))

    if student_name is None:
        run_name = file_name.split('/')[-1].split('.')[0]
    else: 
        run_name = student_name
    save_path = os.path.join(output_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    sft_config = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=4096,
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="Path to CSV file with 'question' and 'answer' columns", default="/home/ubuntu/AcquisitionSynthesis/generating_data/ins_train/random_OpenR1-Math-220k_1000.parquet")
    parser.add_argument("--model_name",  default="Qwen/Qwen2.5-3b-Instruct")
    parser.add_argument("--output_dir", default="/dev/shm/sft_models/", help="Directory to save trained model")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--student_name", type=str, default=None)
    parser.add_argument("--num_data", type=int, default=1000)
    args = parser.parse_args()

    model_path = sft_train(args.file_name, args.model_name, args.num_epochs, args.output_dir, args.student_name, args.num_data)