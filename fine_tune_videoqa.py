import os
import random
import numpy as np
import pandas as pd
import torch
from textwrap import dedent
from typing import Dict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TrainingArguments,
)
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
    SFTTrainer,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
import json
import datetime
import shutil

def get_config():
    """
    Return a dictionary of hyperparameters and file paths.
    Adjust paths as needed for your environment.
    """
    return {
        "SEED": 42,
        "PAD_TOKEN": "<|pad|>",
        "MODEL_NAME": "/mnt/st1/meta-llama/Llama-3.1-8B-Instruct",
        "NEW_MODEL": "./models/Llama-3.1-8B-Instruct-tuned",
        # Where to save checkpoints
        "OUTPUT_DIR": "experiments/checkpoints_16bitLoRA_nonweekonly",
        # Sampling limit for training data
        "TRAIN_DATA_SIZE": 236,
        # Max token length
        "MAX_TOKEN_LENGTH": 8192,
        # File path to save predictions
        "PREDICTION_SAVE_PATH": "./outputs/predictions.csv",
        # Data directories
        "CAPTION_FOLDER": "/mnt/st2/dir_ikeda/panasonic/HCQA/LaViLa_cap5_subset/",
        "SUMMARY_FOLDER": "/mnt/st2/dir_ikeda/panasonic/HCQA/summary_subset",
        "QUESTION_JSON": "/mnt/st2/dir_ikeda/panasonic/HCQA/questions.json",
        "SUBSET_ANSWERS_JSON": "/mnt/st2/dir_ikeda/panasonic/HCQA/subset_answers.json",
        # Path to incorrect_uids.txt
        "INCORRECT_UIDS_PATH": "/mnt/st1/dir_ikeda/panasonic/llm_finetune/ego_schema_qa-dg/incorrect_uids.txt",
        # Max memory settings if multiple GPUs are used in a single process
        "MAX_MEMORY": {
            0: "40GiB",
            1: "40GiB",
            2: "40GiB",
            3: "40GiB",
            # 4: "40GiB",
            # 5: "40GiB",
        },
    }

def seed_everything(seed: int):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in a given text using the specified tokenizer.
    """
    return len(
        tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    )

def format_example(row: dict, tokenizer) -> str:
    """
    Prepare a formatted text prompt for training, including:
    - system prompt
    - question
    - 5 options
    - answer appended at the end
    """
    system_prompt = dedent("""\
        You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [CAPTION], [SUMMARY], [QUESTION]. Where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
    """).strip()
    
    instruction_header = dedent("""\
        [CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each line represents five captions of a 4s video clip, each caption is separated by a semicolon, with a total of 45 lines describing 180 seconds of video. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
        [SUMMARY]: Based on the CAPTIONS of these video clips, an overall description of the video, in chronological order.
        [QUESTION]: A question about video that needs to be answered.
        [OPTION]: Five candidates for the question.
        Now, you should choose the correct option as the answer.
    """).strip()
    
    dynamic_content = dedent(f"""\
        [CAPTION]:
        {row["captions"]}
        [SUMMARY]:
        {row["context"]}
        [QUESTION]:
        {row["question"]}
        [OPTION]:
        {row["options"]}
        [ANSWER]:
    """).strip()
    
    full_prompt = f"{system_prompt}\n{instruction_header}\n{dynamic_content}\n{row['answer']}"
    return full_prompt

def create_test_prompt(data_row, tokenizer) -> str:
    """
    Create a prompt for inference without appending the answer.
    """
    system_prompt = dedent("""\
        You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [CAPTION], [SUMMARY], [QUESTION]. Where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
    """).strip()
    
    instruction_header = dedent("""\
        [CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each line represents five captions of a 4s video clip, each caption is separated by a semicolon, with a total of 45 lines describing 180 seconds of video. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
        [SUMMARY]: Based on the CAPTIONS of these video clips, an overall description of the video, in chronological order.
        [QUESTION]: A question about video that needs to be answered.
        [OPTION]: Five candidates for the question.
        Now, you should choose the correct option as the answer.
    """).strip()
    
    dynamic_content = dedent(f"""\
        [CAPTION]:
        {data_row["captions"]}
        [SUMMARY]:
        {data_row["context"]}
        [QUESTION]:
        {data_row["question"]}
        [OPTION]:
        {data_row["options"]}
        [ANSWER]:
    """).strip()
    
    full_prompt = f"{system_prompt}\n{instruction_header}\n{dynamic_content}"
    return full_prompt

def calculate_accuracy(df, answer_col="answer", prediction_col="prediction"):
    """
    Calculate accuracy by comparing the 'answer' column with the 'prediction' column.
    """
    correct_answers = df[answer_col].astype(str).str.strip()
    model_predictions = df[prediction_col].astype(str).str.strip()
    return accuracy_score(correct_answers, model_predictions) * 100

def load_incorrect_uids(path: str):
    """
    Load a list of UIDs from a file (incorrect_uids.txt) and return as a set of strings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"UID list file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        uid_set = set(line.strip() for line in lines if line.strip())
    return uid_set

def load_and_preprocess_dataset(config, tokenizer):
    """
    1. Read question data from JSON.
    2. Read subset answers from JSON.
    3. Filter based on subset answers, remove any items that do not match.
    4. Exclude UIDs listed in incorrect_uids.txt (depending on your specific training objective).
    5. Collect data (captions, summary, question, options, answer) and build DataFrame.
    6. Generate text prompts, count tokens, and filter based on MAX_TOKEN_LENGTH.
    7. Split into train/val/test sets.
    """
    incorrect_uids = load_incorrect_uids(config["INCORRECT_UIDS_PATH"])

    with open(config["QUESTION_JSON"], "r", encoding="utf-8") as f:
        question_data = json.load(f)
    with open(config["SUBSET_ANSWERS_JSON"], "r", encoding="utf-8") as f:
        subset_answers = json.load(f)

    rows = []
    for item in question_data:
        q_uid = item.get("q_uid")
        
        # Exclude if not in subset_answers
        if q_uid not in subset_answers:
            continue
        
        # Example usage of incorrect_uids:
        # This script *excludes* those in the incorrect_uids set
        if q_uid in incorrect_uids:
            continue

        correct_index = subset_answers[q_uid]
        answer_option_key = f"option {correct_index}"
        if answer_option_key not in item:
            continue

        answer_text = item[answer_option_key]
        question_text = item["question"]

        summary_path = os.path.join(config["SUMMARY_FOLDER"], f"{q_uid}.txt")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r", encoding="utf-8") as summ_f:
            context_text = summ_f.read().strip()

        caption_path = os.path.join(config["CAPTION_FOLDER"], f"{q_uid}.json")
        if not os.path.exists(caption_path):
            continue
        with open(caption_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)

        caps = ''
        for c in captions:
            caps += c['Caption'] + "\n"

        # Concatenate the 5 options into one string
        options_str = ""
        for i in range(5):
            option_key = f"option {i}"
            if option_key in item:
                options_str += f"option {i}: {item[option_key]}\n"

        rows.append({
            "question": question_text,
            "captions": caps,
            "context": context_text,
            "answer": answer_text,
            "options": options_str
        })

    if len(rows) == 0:
        raise ValueError(
            "No matched data found after filtering by incorrect_uids. "
            "Check your summary/question/subset_answers files or UIDs."
        )

    df = pd.DataFrame(rows)
    df["text"] = df.apply(lambda row: format_example(row, tokenizer), axis=1)
    df["token_count"] = df["text"].apply(lambda x: count_tokens(x, tokenizer))
    df = df[df["token_count"] < config["MAX_TOKEN_LENGTH"]]
    
    sample_size = min(len(df), config["TRAIN_DATA_SIZE"])
    df = df.sample(sample_size, random_state=config["SEED"])
    
    print(f"Remaining samples after filtering and sampling: {len(df)}")

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=config["SEED"])
    val_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=config["SEED"])

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    return train_dataset, val_dataset, test_dataset, test_df

def load_base_model_16bit(config, tokenizer):
    """
    Load the base model in 16-bit precision for both inference and training.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        config["MODEL_NAME"],
        device_map="auto",
        max_memory=config["MAX_MEMORY"],
        torch_dtype=torch.float16,  # Load in 16-bit
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    base_model.config.use_cache = False
    return base_model

def train_lora_sft(config, base_model, train_dataset, val_dataset, tokenizer):
    """
    Configure LoRA and train the model using SFTTrainer.
    """
     # Ensure parameters require gradients
    for param in base_model.parameters():
        param.requires_grad = False  # Set all params to not require gradient

    # Print the model structure to identify the correct target modules
    print("Model structure:")
    for name, module in base_model.named_modules():
        if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "gate_proj" in name:
            print(f"Found target: {name}")


    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj", 
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)

    # Verify trainable parameters
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_param_count}")

    num_epochs = 20
    sft_config = SFTConfig(
        output_dir=config["OUTPUT_DIR"],
        dataset_text_field="text",
        max_seq_length=8192,
        num_train_epochs=num_epochs,
        max_steps=-1,  # Use the number of epochs instead
        
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        
        save_strategy="steps",
        save_steps=20,
        save_total_limit=300,
        
        optim="paged_adamw_8bit",
        eval_strategy="epoch",
        logging_steps=10,
        learning_rate=1e-5,
        fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        save_safetensors=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        seed=config["SEED"],
        gradient_checkpointing=False,
    )

    response_template = "[ANSWER]:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.model.config.use_cache = False

    # Training
    trainer.train()

    # Save final model
    trainer.save_model(config["NEW_MODEL"])
    return trainer

def merge_lora_model(config, tokenizer):
    """
    Merge LoRA weights into the base model to produce a standalone final model.
    """
    merged_base_model = AutoModelForCausalLM.from_pretrained(
        config["MODEL_NAME"],
        device_map="auto",
        max_memory=config["MAX_MEMORY"],
        torch_dtype=torch.float16,
        local_files_only=True
    )
    merged_base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    merged_base_model.config.use_cache = True

    peft_model = PeftModel.from_pretrained(merged_base_model, config["NEW_MODEL"])
    final_model = peft_model.merge_and_unload()
    return final_model

def inference_and_evaluation(config, final_model, test_df, tokenizer, trainer):
    """
    1. Perform inference with the final model on the test set.
    2. Calculate accuracy.
    3. Also evaluate with trainer's eval method if available.
    """
    infer_pipe = pipeline(
        task="text-generation",
        model=final_model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        return_full_text=False,
    )

    predictions = []
    for _, row in test_df.iterrows():
        prompt = create_test_prompt(row, tokenizer)
        output = infer_pipe(prompt)[0]["generated_text"]
        predictions.append(output.strip())

    test_df["prediction"] = predictions
    acc = calculate_accuracy(test_df, answer_col="answer", prediction_col="prediction")

    trainer_eval_metrics = trainer.evaluate()
    test_df.to_csv(config["PREDICTION_SAVE_PATH"], index=False)

    return acc, trainer_eval_metrics

def main():
    config = get_config()
    seed_everything(config["SEED"])

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"], local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": config["PAD_TOKEN"]})
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ANSWER]:"]})

    # Load and preprocess data
    train_dataset, val_dataset, test_dataset, test_df = load_and_preprocess_dataset(config, tokenizer)

    # Load base model in 16-bit
    base_model = load_base_model_16bit(config, tokenizer)

    # Train LoRA (SFT)
    trainer = train_lora_sft(config, base_model, train_dataset, val_dataset, tokenizer)

    # Merge LoRA into base model
    final_model = merge_lora_model(config, tokenizer)

    # Inference and evaluation
    acc, trainer_eval_metrics = inference_and_evaluation(config, final_model, test_df, tokenizer, trainer)
    print(f"Test Accuracy: {acc:.2f}%")

    print("Complete all process.")

if __name__ == "__main__":
    main()
    print("Done.")