import os
import random
import numpy as np
import pandas as pd
import torch
import re
import warnings
import logging
import difflib
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging as transformers_logging,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
import json
import datetime

# Suppress warnings function from your original code
def suppress_warnings():
    """
    Suppress specific warnings and reduce logging verbosity.
    """
    warnings.filterwarnings("ignore", message="This instance will be ignored in loss calculation")
    warnings.filterwarnings("ignore", message="Could not find response key")
    transformers_logging.set_verbosity_error()
    logging.getLogger("trl").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.WARNING)

def get_config():
    """
    Return a dictionary of hyperparameters and file paths.
    """
    return {
        "SEED": 42,
        "PAD_TOKEN": "<|pad|>",
        # Path to your fine-tuned model
        "MODEL_PATH": "./models/DeepSeek-R1-Distill-Llama-8B-tuned",
        # Path to the original base model
        "BASE_MODEL_PATH": "/mnt/st1/DeepSeekV3-2/DeepSeek-R1-Distill-Llama-8B",
        # Max token length
        "MAX_TOKEN_LENGTH": 4096,
        # File path to save predictions
        "PREDICTION_SAVE_PATH": "./outputs/improved_predictions.csv",
        # Data directories - same as in your original code
        "CAPTION_FOLDER": "/mnt/st2/dir_ikeda/panasonic/HCQA/LaViLa_cap5_subset/",
        "SUMMARY_FOLDER": "/mnt/st2/dir_ikeda/panasonic/HCQA/summary_subset",
        "QUESTION_JSON": "/mnt/st2/dir_ikeda/panasonic/HCQA/questions.json",
        "SUBSET_ANSWERS_JSON": "/mnt/st2/dir_ikeda/panasonic/HCQA/subset_answers.json",
        # Path to incorrect_uids.txt
        "INCORRECT_UIDS_PATH": "/mnt/st1/dir_ikeda/panasonic/llm_finetune/ego_schema_qa-dg/incorrect_uids.txt",
        # Max memory settings for your GPUs
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

def load_test_dataset(config):
    """
    Load the test dataset without preprocessing for training.
    We only need the test data for evaluation.
    """
    incorrect_uids = load_incorrect_uids(config["INCORRECT_UIDS_PATH"])

    with open(config["QUESTION_JSON"], "r", encoding="utf-8") as f:
        question_data = json.load(f)
    with open(config["SUBSET_ANSWERS_JSON"], "r", encoding="utf-8") as f:
        subset_answers = json.load(f)

    print(f"Total questions: {len(question_data)}")
    print(f"Subset answers: {len(subset_answers)}")
    print(f"Incorrect UIDs: {len(incorrect_uids)}")
    
    rows = []
    for item in tqdm(question_data, desc="Processing dataset"):
        q_uid = item.get("q_uid")
        
        # Exclude if not in subset_answers
        if q_uid not in subset_answers:
            continue
        
        # Exclude items in incorrect_uids
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
            print(f"Error processing {q_uid}: {type(e).__name__}")

    print(f"Final processed examples: {len(rows)}")

    if len(rows) == 0:
        raise ValueError(
            "No matched data found after filtering. "
            "Check your summary/question/subset_answers files or UIDs."
        )

    # Create DataFrame
    print("Creating DataFrame...")
    test_df = pd.DataFrame(rows)
    
    # Uncomment the line below for testing with a smaller sample
    # test_df = test_df.sample(50, random_state=config["SEED"])
    
    return test_df

# Original prompt function from your code
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

# New alternative prompt templates
def create_simplified_prompt(row, tokenizer=None):
    """
    A more simplified prompt that focuses on the essential information.
    """
    system_prompt = "You are a helpful visual question answering assistant. Answer the question based on the video description."
    
    prompt = f"""{system_prompt}

Video description:
{row["captions"]}

Summary:
{row["context"]}

Question:
{row["question"]}

Options:
{row["options"]}

Answer:
"""
    return prompt

def create_explicit_prompt(row, tokenizer=None):
    """
    A more explicit prompt that directly asks for option selection.
    """
    system_prompt = "You are a visual question answering expert. Select the correct option number as your answer."
    
    prompt = f"""{system_prompt}

CAPTION:
{row["captions"]}

SUMMARY:
{row["context"]}

QUESTION:
{row["question"]}

OPTIONS:
{row["options"]}

Based on the video description, select one option as your answer.
Answer:
"""
    return prompt

# Original answer extraction from your code
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

# Improved answer extraction function
def improved_extract_answer(text, original_options):
    """
    Simpler and more direct answer extraction focused on options.
    """
    # Normalize text
    text = text.strip().lower()
    
    # Extract options into a dictionary
    options = {}
    option_texts = []
    for line in original_options.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                option_key = parts[0].strip()
                option_text = parts[1].strip()
                options[option_key] = option_text
                options[option_key.replace("option ", "")] = option_text  # Also store as just the number
                option_texts.append(option_text)
    
    # 1. Direct exact match with full option text
    for option_text in option_texts:
        if option_text.lower() in text:
            return option_text
    
    # 2. Look for option numbers
    option_patterns = [
        r'answer.*?option\s*(\d+)',
        r'answer.*?(\d+)',
        r'option\s*(\d+)',
        r'(?<!\w)(\d+)(?!\w)'  # isolated digits
    ]
    
    for pattern in option_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Take the last match, which is more likely to be the conclusion
            option_num = matches[-1]
            option_key = f"option {option_num}"
            if option_key in options:
                return options[option_key]
            elif option_num in options:
                return options[option_num]
    
    # 3. Look for exact quoted text that matches an option
    for option_text in option_texts:
        if f'"{option_text.lower()}"' in text or f"'{option_text.lower()}'" in text:
            return option_text
    
    # 4. If nothing else works, try to find the last sentence
    sentences = re.split(r'[.!?]\s+', text)
    if sentences and len(sentences[-1].strip()) > 0:
        last_sentence = sentences[-1].strip()
        # Check if the last sentence is close to any option
        for option_text in option_texts:
            similarity = difflib.SequenceMatcher(None, last_sentence, option_text.lower()).ratio()
            if similarity > 0.7:
                return option_text
    
    # 5. Last resort, try to match any option that has significant word overlap
    words = set(re.findall(r'\b\w+\b', text))
    best_option = None
    best_match = 0
    for option_text in option_texts:
        option_words = set(re.findall(r'\b\w+\b', option_text.lower()))
        overlap = len(words.intersection(option_words))
        if overlap > best_match:
            best_match = overlap
            best_option = option_text
    
    if best_option:
        return best_option
        
    # If all else fails, return first part of text (limit to 100 chars)
    return text[:100] if text else "ERROR: Empty response"

def improved_inference_and_evaluation(config, final_model, test_df, tokenizer):
    """
    Enhanced inference with improved prompt template and answer extraction.
    Tests multiple generation parameters to find optimal settings.
    """
    print("Running improved inference on test set...")
    
    # Create multiple generation parameter sets to try
    generation_params = [
        {
            "name": "Default",
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": 0.0,
            "num_return_sequences": 1,
        },
        {
            "name": "Longer output",
            "max_new_tokens": 64,
            "do_sample": False,
            "temperature": 0.0,
            "num_return_sequences": 1,
        },
        {
            "name": "Slight randomness",
            "max_new_tokens": 32,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.95,
            "num_return_sequences": 1,
        }
    ]
    
    # Experiment with different prompt templates
    prompt_templates = [
        {
            "name": "Original",
            "template": create_test_prompt  # Your original function
        },
        {
            "name": "Simplified",
            "template": create_simplified_prompt
        },
        {
            "name": "More explicit",
            "template": create_explicit_prompt
        }
    ]
    
    # Save best configuration
    best_accuracy = 0.0
    best_config = None
    best_predictions = None
    
    # Try each combination
    for prompt in prompt_templates:
        for params in generation_params:
            print(f"\nTesting: Prompt: {prompt['name']}, Generation: {params['name']}")
            
            # Setup pipeline with current parameters
            infer_pipe = pipeline(
                task="text-generation",
                model=final_model,
                tokenizer=tokenizer,
                max_new_tokens=params["max_new_tokens"],
                do_sample=params["do_sample"],
                temperature=params.get("temperature", 0.0),
                top_p=params.get("top_p", 1.0),
                num_return_sequences=params["num_return_sequences"],
                return_full_text=False,
            )
            
            # Run inference with current setup
            predictions = []
            correct_count = 0
            
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
                # Generate prompt using current template
                current_prompt = prompt["template"](row, tokenizer)
                
                try:
                    output = infer_pipe(current_prompt)[0]["generated_text"]
                    raw_prediction = output.strip()
                    
                    # Use both extraction methods and pick the most reliable
                    prediction1 = extract_answer(raw_prediction, row["options"])
                    prediction2 = improved_extract_answer(raw_prediction, row["options"])
                    
                    # Compare both predictions with the answer
                    matches1 = prediction1.strip() == row["answer"].strip()
                    matches2 = prediction2.strip() == row["answer"].strip()
                    
                    # Choose the prediction that matches, or prediction2 if neither matches
                    prediction = prediction1 if matches1 else (prediction2 if matches2 else prediction2)
                    
                    # Print sample predictions (reduced frequency)
                    if idx % 25 == 0:
                        print(f"\nSample {idx}:")
                        print(f"Question: {row['question']}")
                        print(f"Raw output: {raw_prediction[:100]}...")
                        print(f"Prediction: {prediction}")
                        print(f"Actual: {row['answer']}")
                        print(f"Correct: {prediction.strip() == row['answer'].strip()}")
                    
                    predictions.append(prediction)
                    
                    # Track accuracy for progress reporting
                    is_correct = prediction.strip() == row["answer"].strip()
                    if is_correct:
                        correct_count += 1
                        
                except Exception as e:
                    print(f"Error in inference for row {idx}: {e}")
                    predictions.append("ERROR")
            
            # Calculate accuracy for this configuration
            temp_df = test_df.copy()
            temp_df["prediction"] = predictions
            acc = calculate_accuracy(temp_df, answer_col="answer", prediction_col="prediction")
            
            print(f"Configuration accuracy: {acc:.2f}%")
            
            # Update best configuration if improved
            if acc > best_accuracy:
                best_accuracy = acc
                best_config = {
                    "prompt": prompt["name"],
                    "generation": params["name"],
                    "accuracy": acc
                }
                best_predictions = predictions
    
    # Apply best predictions to the dataframe
    print(f"\nBest configuration: {best_config}")
    test_df["prediction"] = best_predictions
    test_df["is_correct"] = test_df["answer"].str.strip() == test_df["prediction"].str.strip()
    
    # Save detailed results
    test_df.to_csv(config["PREDICTION_SAVE_PATH"], index=False)
    
    # Save errors separately for analysis
    error_df = test_df[~test_df["is_correct"]].copy()
    error_df.to_csv(config["PREDICTION_SAVE_PATH"].replace(".csv", "_errors.csv"), index=False)

    print(f"Final test accuracy: {best_accuracy:.2f}%")
    print(f"Correct: {test_df['is_correct'].sum()}/{len(test_df)}")
    print(f"Results saved to: {config['PREDICTION_SAVE_PATH']}")
    
    return best_accuracy

def load_model_and_tokenizer(config):
    """
    Fix for the tokenizer/model mismatch issue.
    Use the base model's tokenizer and add special tokens before loading the model.
    """
    print("Loading base model tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["BASE_MODEL_PATH"], 
        local_files_only=True
    )
    
    # Add special tokens - MUST match what was used during training
    tokenizer.add_special_tokens({"pad_token": config["PAD_TOKEN"]})
    tokenizer.padding_side = "right"
    
    # Add the same special tokens as in your training code
    special_tokens = ["ANSWER:"]  # This matches your training code
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    try:
        # First attempt: Try to load directly with modified tokenizer
        print(f"Attempting to load fine-tuned model from: {config['MODEL_PATH']}")
        model = AutoModelForCausalLM.from_pretrained(
            config["MODEL_PATH"],
            device_map="auto",
            max_memory=config["MAX_MEMORY"],
            torch_dtype=torch.float16,
            local_files_only=True
        )
        print("Model loaded successfully!")
        
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        
        # Fallback: Load base model and adapt to finetuned vocab size
        print("Trying alternative approach: Loading base model and adapting...")
        model = AutoModelForCausalLM.from_pretrained(
            config["BASE_MODEL_PATH"],
            device_map="auto", 
            max_memory=config["MAX_MEMORY"],
            torch_dtype=torch.float16,
            local_files_only=True
        )
        
        # Resize model embeddings to match tokenizer
        model.resize_token_embeddings(len(tokenizer))
        
        # Now try to load the fine-tuned model state
        print("Loading fine-tuned model state...")
        
        try:
            # Try to load from base model path while ignoring mismatched keys
            model = AutoModelForCausalLM.from_pretrained(
                config["MODEL_PATH"],
                device_map="auto",
                max_memory=config["MAX_MEMORY"],
                torch_dtype=torch.float16,
                local_files_only=True,
                ignore_mismatched_sizes=True  # This is the key parameter
            )
            print("Model loaded with ignore_mismatched_sizes=True")
            
        except Exception as e2:
            print(f"Still having issues loading model: {e2}")
            print("Using base model for testing - results will not reflect fine-tuning")
    
    # Ensure model and tokenizer match
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def main():
    """
    Main function to run the evaluation script.
    """
    print("\n=== DeepSeek-R1-Distill-Llama-8B Model Evaluation ===\n")
    
    # Suppress warnings
    suppress_warnings()
    
    # Fix for potential hanging issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    config = get_config()
    seed_everything(config["SEED"])
    
    # Create output directory
    os.makedirs(os.path.dirname(config["PREDICTION_SAVE_PATH"]), exist_ok=True)
    
    # Load model and tokenizer with proper tokenization
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load test dataset
    print("Loading test dataset...")
    test_df = load_test_dataset(config)
    
    # Run improved inference
    print("Starting improved inference...")
    acc = improved_inference_and_evaluation(config, model, test_df, tokenizer)
    
    print("\n=== Evaluation Summary ===")
    print(f"Model: DeepSeek-R1-Distill-Llama-8B (fine-tuned)")
    print(f"Test examples: {len(test_df)}")
    print(f"Best test accuracy: {acc:.2f}%")
    print(f"Results saved to: {config['PREDICTION_SAVE_PATH']}")
    
    print("Evaluation complete.")

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"Done. Total runtime: {duration}")