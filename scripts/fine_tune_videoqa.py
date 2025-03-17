import os
import random
import numpy as np
import pandas as pd
import torch
import re
import warnings
import logging
import difflib  # Added for similarity matching
from textwrap import dedent
from typing import Dict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TrainingArguments,
    logging as transformers_logging,
    TrainerCallback,  # Added for custom callback
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

def suppress_warnings():
    """
    Suppress specific warnings and reduce logging verbosity.
    """
    # Filter out specific warnings
    warnings.filterwarnings("ignore", message="This instance will be ignored in loss calculation")
    warnings.filterwarnings("ignore", message="Could not find response key")
    
    # Reduce transformers logging
    transformers_logging.set_verbosity_error()
    
    # Reduce TRL logging
    logging.getLogger("trl").setLevel(logging.ERROR)
    
    # Reduce other loggers
    logging.getLogger("peft").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.WARNING)

def truncate_caption(text, max_length=50):
    """Truncate long text for logging purposes."""
    if text and len(text) > max_length:
        return text[:max_length] + "... [truncated]"
    return text

def get_config():
    """
    Return a dictionary of hyperparameters and file paths.
    """
    return {
        "SEED": 42,
        "PAD_TOKEN": "<|pad|>",
        "MODEL_NAME": "/mnt/st1/DeepSeekV3-2/DeepSeek-R1-Distill-Llama-8B",
        "NEW_MODEL": "./models/DeepSeek-R1-Distill-Llama-8B-tuned",
        # Where to save checkpoints
        "OUTPUT_DIR": "experiments/checkpoints_16bitLoRA_DeepSeek",
        # Sampling limit for training data - reduced to improve performance
        "TRAIN_DATA_SIZE": 200,  # Reduced from 500 to 200 for better focus
        # Max token length
        "MAX_TOKEN_LENGTH": 4096,
        # File path to save predictions
        "PREDICTION_SAVE_PATH": "./outputs/deepseek_predictions.csv",
        # Data directories
        "CAPTION_FOLDER": "/mnt/st2/dir_ikeda/panasonic/HCQA/LaViLa_cap5_subset/",
        "SUMMARY_FOLDER": "/mnt/st2/dir_ikeda/panasonic/HCQA/summary_subset",
        "QUESTION_JSON": "/mnt/st2/dir_ikeda/panasonic/HCQA/questions.json",
        "SUBSET_ANSWERS_JSON": "/mnt/st2/dir_ikeda/panasonic/HCQA/subset_answers.json",
        # Path to incorrect_uids.txt
        "INCORRECT_UIDS_PATH": "/mnt/st1/dir_ikeda/panasonic/llm_finetune/ego_schema_qa-dg/incorrect_uids.txt",
        # Max memory settings for 8 RTX A6000 GPUs
        "MAX_MEMORY": {
            0: "0GiB",     # Limited as this GPU has 44GB already in use
            1: "42GiB",    # Almost entirely free
            2: "42GiB",    # Almost entirely free
            3: "42GiB",    # Almost entirely free
            4: "42GiB",    # Almost entirely free
            5: "42GiB",    # Almost entirely free
            6: "42GiB",    # Almost entirely free
            7: "42GiB",    # Almost entirely free
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
    Improved prompt with better examples for video QA.
    """
    system_prompt = "You are a helpful visual question answering assistant specifically trained to understand and answer questions about videos."
    
    # More relevant few-shot examples for video QA
    few_shot = """
Example 1:
Question: What activity is the person primarily engaged in throughout the video?
Options:
option 0: Playing a musical instrument
option 1: Cooking food in the kitchen
option 2: Reading a book
option 3: Exercising
Answer: Cooking food in the kitchen

Example 2:
Question: What specific object does the person interact with most frequently?
Options:
option 0: Smartphone
option 1: Book
option 2: Kitchen knife
option 3: Remote control
Answer: Kitchen knife

Example 3:
Question: How would you describe the main action performed by the person in this video?
Options:
option 0: The person is cutting vegetables precisely
option 1: The person is typing on a keyboard
option 2: The person is walking around aimlessly
option 3: The person is talking to another individual
Answer: The person is cutting vegetables precisely
"""
    
    # Simple, clear instruction
    dynamic_content = f"""
CAPTION:
{row["captions"]}

SUMMARY:
{row["context"]}

QUESTION:
{row["question"]}

OPTIONS:
{row["options"]}

Answer:
"""
    
    full_prompt = f"{system_prompt}\n\n{few_shot}\n\n{dynamic_content}{row['answer']}"
    return full_prompt

def create_test_prompt(data_row, tokenizer) -> str:
    """
    Create test prompt with improved examples for video QA.
    """
    system_prompt = "You are a helpful visual question answering assistant specifically trained to understand and answer questions about videos."
    
    # More relevant few-shot examples for video QA
    few_shot = """
Example 1:
Question: What activity is the person primarily engaged in throughout the video?
Options:
option 0: Playing a musical instrument
option 1: Cooking food in the kitchen
option 2: Reading a book
option 3: Exercising
Answer: Cooking food in the kitchen

Example 2:
Question: What specific object does the person interact with most frequently?
Options:
option 0: Smartphone
option 1: Book
option 2: Kitchen knife
option 3: Remote control
Answer: Kitchen knife

Example 3:
Question: How would you describe the main action performed by the person in this video?
Options:
option 0: The person is cutting vegetables precisely
option 1: The person is typing on a keyboard
option 2: The person is walking around aimlessly
option 3: The person is talking to another individual
Answer: The person is cutting vegetables precisely
"""
    
    # Simple, clear instruction with full captions for model
    dynamic_content = f"""
CAPTION:
{data_row["captions"]}

SUMMARY:
{data_row["context"]}

QUESTION:
{data_row["question"]}

OPTIONS:
{data_row["options"]}

Answer:
"""
    
    full_prompt = f"{system_prompt}\n\n{few_shot}\n\n{dynamic_content}"
    return full_prompt

def extract_answer(text, original_options):
    """
    Enhanced answer extraction with more robust pattern matching.
    """
    # Clean up text by removing common prefixes
    cleaned_text = re.sub(r'^(the user|this appears|i think|i need|let me|based on|according to|okay|alright|let\'s)', '', text.lower(), flags=re.IGNORECASE)
    cleaned_text = cleaned_text.strip()
    
    # Extract option texts from original options
    option_texts = []
    option_dict = {}
    for line in original_options.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                option_num = parts[0].strip()
                option_text = parts[1].strip()
                option_texts.append(option_text)
                option_dict[option_num] = option_text
                # Also add number-only keys for backup matching
                if "option" in option_num:
                    num_only = option_num.replace("option", "").strip()
                    option_dict[num_only] = option_text
    
    # NEW: Direct exact match with the complete answer (best case)
    for option_text in option_texts:
        if cleaned_text == option_text.lower():
            return option_text
    
    # Try several extraction methods in order of reliability
    
    # 1. Direct match with option text (partial)
    for option_text in sorted(option_texts, key=len, reverse=True):
        if option_text.lower() in cleaned_text:
            return option_text
        # NEW: Also try if cleaned text is in option (for short answers)
        if len(cleaned_text) > 10 and cleaned_text in option_text.lower():
            return option_text
    
    # 2. Look for option number patterns (improved with more patterns)
    option_patterns = [
        r'option\s*(\d+)',
        r'answer\s*(?:is|:)?\s*option\s*(\d+)',
        r'(?:choose|select|pick)\s*option\s*(\d+)',
        r'(?:^|\s)(\d+)(?:$|\s)',  # standalone digit
        r'answer\s*(?:is|:)?\s*(\d+)',  # "answer: 2"
        r'(?:would be|should be|is)\s*(\d+)'  # "would be 2"
    ]
    
    for pattern in option_patterns:
        option_match = re.search(pattern, cleaned_text)
        if option_match:
            option_num = option_match.group(1)
            option_key = f"option {option_num}"
            if option_key in option_dict:
                return option_dict[option_key]
            elif option_num in option_dict:  # Try number-only key
                return option_dict[option_num]
    
    # 3. Check key terms from each option
    for option_text in option_texts:
        # Count distinctive words from each option in the answer
        words = option_text.lower().split()
        distinctive_words = [w for w in words if len(w) > 3]  # Only meaningful words
        
        # If more than half of distinctive words are in the answer, it's likely the right one
        matches = sum(1 for word in distinctive_words if word in cleaned_text)
        if distinctive_words and matches >= len(distinctive_words) / 2:
            return option_text
    
    # 4. Look for key nouns from the options in the text
    for option_text in option_texts:
        nouns = [word for word in option_text.lower().split() if len(word) > 3]
        for noun in nouns:
            if noun in cleaned_text:
                return option_text
    
    # 5. If all else fails, try to find a direct quote
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines:
        for option_text in option_texts:
            # Compute similarity ratio
            similarity = difflib.SequenceMatcher(None, line.lower(), option_text.lower()).ratio()
            if similarity > 0.7:  # If more than 70% similar
                return option_text
    
    # 6. Last resort: find the shortest line that's not a meta-comment
    valid_lines = [line for line in lines if not line.lower().startswith(('the user', 'this appears', 'i think'))]
    if valid_lines:
        best_line = min(valid_lines, key=len)
        if len(best_line) < 100:
            return best_line
    
    # If nothing works, just return a clean version of the first line
    return text.split('\n')[0].strip()[:100] if text else "ERROR: Empty response"

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
    Load and preprocess the dataset with improved error tracking and reduced logging.
    Never prints captions to logs.
    """
    incorrect_uids = load_incorrect_uids(config["INCORRECT_UIDS_PATH"])

    with open(config["QUESTION_JSON"], "r", encoding="utf-8") as f:
        question_data = json.load(f)
    with open(config["SUBSET_ANSWERS_JSON"], "r", encoding="utf-8") as f:
        subset_answers = json.load(f)

    print(f"Total questions: {len(question_data)}")
    print(f"Subset answers: {len(subset_answers)}")
    print(f"Incorrect UIDs: {len(incorrect_uids)}")

    # Track filtering statistics
    stats = {
        "total": len(question_data),
        "not_in_subset": 0,
        "in_incorrect_uids": 0,
        "missing_option": 0,
        "missing_summary": 0,
        "missing_caption": 0
    }
    
    rows = []
    for item in tqdm(question_data, desc="Processing dataset"):
        q_uid = item.get("q_uid")
        
        # Exclude if not in subset_answers
        if q_uid not in subset_answers:
            stats["not_in_subset"] += 1
            continue
        
        # Exclude items in incorrect_uids
        if q_uid in incorrect_uids:
            stats["in_incorrect_uids"] += 1
            continue

        correct_index = subset_answers[q_uid]
        answer_option_key = f"option {correct_index}"
        if answer_option_key not in item:
            stats["missing_option"] += 1
            continue

        answer_text = item[answer_option_key]
        question_text = item["question"]

        summary_path = os.path.join(config["SUMMARY_FOLDER"], f"{q_uid}.txt")
        if not os.path.exists(summary_path):
            stats["missing_summary"] += 1
            continue

        with open(summary_path, "r", encoding="utf-8") as summ_f:
            context_text = summ_f.read().strip()

        caption_path = os.path.join(config["CAPTION_FOLDER"], f"{q_uid}.json")
        if not os.path.exists(caption_path):
            stats["missing_caption"] += 1
            continue
            
        try:
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
                "q_uid": q_uid,  # Add UID for tracking
                "question": question_text,
                "captions": caps,
                "context": context_text,
                "answer": answer_text,
                "options": options_str,
                "correct_index": correct_index  # Store index for analysis
            })
        except Exception as e:
            # Don't print the exception details with caption data
            print(f"Error processing {q_uid}: {type(e).__name__}")

    print(f"Filtering statistics: {stats}")
    print(f"Final processed examples: {len(rows)}")

    if len(rows) == 0:
        raise ValueError(
            "No matched data found after filtering. "
            "Check your summary/question/subset_answers files or UIDs."
        )

    # Create DataFrame and format examples
    print("Creating DataFrame and formatting examples...")
    df = pd.DataFrame(rows)
    df["text"] = df.apply(lambda row: format_example(row, tokenizer), axis=1)
    df["token_count"] = df["text"].apply(lambda x: count_tokens(x, tokenizer))
    
    # Filter by token length
    orig_len = len(df)
    df = df[df["token_count"] < config["MAX_TOKEN_LENGTH"]]
    print(f"Removed {orig_len - len(df)} examples exceeding token limit")
    
    # Sample if needed
    sample_size = min(len(df), config["TRAIN_DATA_SIZE"])
    if sample_size < len(df):
        df = df.sample(sample_size, random_state=config["SEED"])
        print(f"Sampled {sample_size} from {len(df)} valid examples")
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=config["SEED"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=config["SEED"])
    
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save splits for reference but don't print the content
    print("Saving train/val/test splits to CSV...")
    train_df[["q_uid", "question", "answer"]].to_csv("train_split.csv", index=False)
    val_df[["q_uid", "question", "answer"]].to_csv("val_split.csv", index=False)
    test_df[["q_uid", "question", "answer"]].to_csv("test_split.csv", index=False)

    # Convert to Datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    return train_dataset, val_dataset, test_dataset, test_df

def load_base_model_16bit(config, tokenizer):
    """
    Load the base model in 16-bit precision for both inference and training.
    """
    print(f"Loading model: {config['MODEL_NAME']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["MODEL_NAME"],
        device_map="auto",
        max_memory=config["MAX_MEMORY"],
        torch_dtype=torch.float16,  # Load in 16-bit
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    base_model.config.use_cache = False  # Disable KV cache during training
    return base_model

def inspect_model_architecture(model):
    """
    Print the full model architecture to identify correct target modules.
    """
    print("\nDEEPSEEK MODEL ARCHITECTURE ANALYSIS")
    print("=" * 50)
    
    # Track all module names for verification
    module_names = []
    
    for name, _ in model.named_modules():
        module_names.append(name)
        
    # Check for up_proj and down_proj
    up_proj_exists = any("up_proj" in name for name in module_names)
    down_proj_exists = any("down_proj" in name for name in module_names)
    
    print(f"up_proj modules found: {up_proj_exists}")
    print(f"down_proj modules found: {down_proj_exists}")
    
    return up_proj_exists, down_proj_exists

# Custom callback class for better progress reporting
class DetailedProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            current_step = state.global_step
            total_steps = state.max_steps if state.max_steps > 0 else "?"
            epoch = state.epoch if state.epoch is not None else "?"
            step_loss = logs.get("loss", "N/A")
            step_learning_rate = logs.get("learning_rate", "N/A")
            
            print(f"Progress: Step {current_step}/{total_steps} | Epoch {epoch:.2f} | Loss: {step_loss:.4f} | LR: {step_learning_rate:.8f}")
            
            # If eval logs are present, print those too
            if "eval_loss" in logs:
                print(f"Eval Loss: {logs['eval_loss']:.4f}")

def train_lora_sft(config, base_model, train_dataset, val_dataset, tokenizer):
    """
    Configure LoRA and train the model using SFTTrainer.
    Optimized for higher accuracy on video QA tasks.
    """
    # Inspect model to confirm module structure
    print("Inspecting model architecture...")
    up_proj_exists, down_proj_exists = inspect_model_architecture(base_model)
    
    # Set all parameters to not require gradients
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Build target modules list based on model inspection
    target_modules = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
    ]
    
    # Add these if they exist
    if up_proj_exists:
        target_modules.append("mlp.up_proj")
    if down_proj_exists:
        target_modules.append("mlp.down_proj")
    
    # IMPROVED: Higher rank and alpha for better performance
    lora_config = LoraConfig(
        r=16,  # Increased from 8 to 16 for better capacity
        lora_alpha=32,  # Increased from 16 to 32
        target_modules=target_modules,
        lora_dropout=0.1,  # Slightly higher dropout for regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)

    # Verify trainable parameters
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_param_count}")

    # IMPROVED: Training config optimized for video QA
    sft_config = SFTConfig(
        output_dir=config["OUTPUT_DIR"],
        dataset_text_field="text",
        max_seq_length=config["MAX_TOKEN_LENGTH"],
        num_train_epochs=8,  # Increased back to 8 from 5
        max_steps=-1,
        
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        
        save_strategy="epoch",
        save_steps=5,
        save_total_limit=3,
        
        optim="paged_adamw_8bit",
        eval_strategy="epoch",
        logging_steps=10,  # More frequent logging (was 20)
        learning_rate=1e-5,  # Increased from 5e-6 to 1e-5
        weight_decay=0.01,
        fp16=True,
        warmup_ratio=0.05,  # Reduced warmup to 5%
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        save_safetensors=True,
        
        # Keep early stopping but with more patience
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        seed=config["SEED"],
        gradient_checkpointing=False,
    )

    # Use "ANSWER:" as response template for DeepSeek
    response_template = "ANSWER:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[DetailedProgressCallback()],  # Add our custom callback
    )

    trainer.model.config.use_cache = False

    # Training
    print("Starting training with optimized parameters...")
    trainer.train()

    # Save final model
    print("Saving model...")
    trainer.save_model(config["NEW_MODEL"])
    return trainer

def merge_lora_model(config, tokenizer):
    """
    Merge LoRA weights into the base model to produce a standalone final model.
    """
    print("Merging LoRA weights into base model...")
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
    Enhanced inference with improved processing and error analysis.
    Never prints captions to logs.
    """
    print("Running inference on test set...")
    infer_pipe = pipeline(
        task="text-generation",
        model=final_model,
        tokenizer=tokenizer,
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0,
        num_return_sequences=1,
        return_full_text=False,
    )

    # Add progress tracking
    predictions = []
    correct_count = 0
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        # Use full prompt for model inference (but never log it)
        prompt = create_test_prompt(row, tokenizer)
        
        try:
            output = infer_pipe(prompt)[0]["generated_text"]
            raw_prediction = output.strip()
            prediction = extract_answer(raw_prediction, row["options"])
            
            # Only print question and answer info, never caption data
            if idx % 10 == 0:
                print(f"\nSample {idx}:")
                print(f"Question: {row['question']}")
                print(f"Prediction: {prediction}")
                print(f"Actual: {row['answer']}")
                print(f"Correct: {prediction.strip() == row['answer'].strip()}")
            
            predictions.append(prediction)
            
            # Track accuracy for progress reporting
            is_correct = prediction.strip() == row["answer"].strip()
            if is_correct:
                correct_count += 1
                
            # Print occasional accuracy updates
            if idx % 20 == 0:
                print(f"Running accuracy: {correct_count/(idx+1)*100:.2f}%")
                
        except Exception as e:
            print(f"Error in inference for row {idx}: {e}")
            predictions.append("ERROR")

    test_df["prediction"] = predictions
    acc = calculate_accuracy(test_df, answer_col="answer", prediction_col="prediction")
    
    # Save detailed results
    test_df["is_correct"] = test_df["answer"].str.strip() == test_df["prediction"].str.strip()
    test_df.to_csv(config["PREDICTION_SAVE_PATH"], index=False)
    
    # Save errors separately for analysis
    error_df = test_df[~test_df["is_correct"]].copy()
    error_df.to_csv(config["PREDICTION_SAVE_PATH"].replace(".csv", "_errors.csv"), index=False)

    print(f"Final test accuracy: {acc:.2f}%")
    print(f"Correct: {test_df['is_correct'].sum()}/{len(test_df)}")
    print(f"Results saved to: {config['PREDICTION_SAVE_PATH']}")
    print(f"Error analysis saved to: {config['PREDICTION_SAVE_PATH'].replace('.csv', '_errors.csv')}")

    return acc, None  # Fixed return value

def main():
    print("\n=== DeepSeek-R1-Distill-Llama-8B Fine-Tuning for Video QA ===\n")
    
    # Add this line to suppress warnings
    suppress_warnings()
    
    # Add these fixes for potential hanging issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    config = get_config()
    seed_everything(config["SEED"])

    # Create output directories
    os.makedirs(os.path.dirname(config["PREDICTION_SAVE_PATH"]), exist_ok=True)
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"], local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": config["PAD_TOKEN"]})
    tokenizer.padding_side = "right"
    
    # Add required tokens for DeepSeek
    special_tokens = ["ANSWER:"]  # Changed from [ANSWER]: for DeepSeek
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Vocabulary size: {len(tokenizer)}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_dataset, val_dataset, test_dataset, test_df = load_and_preprocess_dataset(config, tokenizer)

    # Load base model in 16-bit
    base_model = load_base_model_16bit(config, tokenizer)

    # Train with DeepSeek-optimized LoRA config
    trainer = train_lora_sft(config, base_model, train_dataset, val_dataset, tokenizer)

    # Merge LoRA into base model
    final_model = merge_lora_model(config, tokenizer)

    # Run inference and evaluation - THIS IS THE TESTING CODE
    print("\n=== Running Final Testing ===\n")
    acc, _ = inference_and_evaluation(config, final_model, test_df, tokenizer, trainer)
    
    # Final summary
    print("\n=== Training Summary ===")
    print(f"Model: DeepSeek-R1-Distill-Llama-8B")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Saved model: {config['NEW_MODEL']}")
    
    # Print detailed error analysis summary
    error_count = len(test_df) - test_df["is_correct"].sum()
    if error_count > 0:
        print(f"\n=== Error Analysis ===")
        print(f"Total errors: {error_count}/{len(test_df)}")
        print(f"Error details saved to: {config['PREDICTION_SAVE_PATH'].replace('.csv', '_errors.csv')}")
    
    print("Complete all process.")

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"Done. Total runtime: {duration}")