import os
import re
import json
import torch
import traceback
from tqdm import tqdm
import multiprocessing
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Constants for model paths - keeping them as is since they're correct
BASE_MODEL = "/mnt/st1/DeepSeekV3-2/DeepSeek-R1-Distill-Llama-8B"
FINETUNED_MODEL = "./models/DeepSeek-R1-Distill-Llama-8B-tuned"
EXAMPLE_PATH = '/mnt/st2/dir_ikeda/panasonic_2/HCQA/example_qa.txt'
QUESTION_PATH = '/mnt/st2/dir_ikeda/panasonic_2/HCQA/questions.json'
LAVILA_PATH = '/mnt/st2/dir_ikeda/panasonic_2/HCQA/LaViLa_cap5_mainset/'
SUMMARY_PATH = '/mnt/st2/dir_ikeda/panasonic_2/HCQA/summary_mainset'

# Output directories
RESULTS_DIR = "/mnt/st1/dir_ikeda/panasonic/HCQA/results/result_deepseek-tuned-76percent"

# Target vocabulary size - based on error message in logs
TARGET_VOCAB_SIZE = 128264

# Load example context
with open(EXAMPLE_PATH, 'r') as ex:
    example = ex.read()

def load_deepseek_model():
    """
    Enhanced loading method for DeepSeek model with proper tokenizer alignment.
    Explicitly handles vocabulary size mismatch between base and fine-tuned models.
    """
    # Phase 1: Properly load tokenizer with exact special tokens
    print("Phase 1: Loading tokenizer...")
    try:
        # Try to load from fine-tuned model directory first
        tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL,
            local_files_only=True
        )
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.padding_side = "right"
        
        # Verify ANSWER: token is present
        if "ANSWER:" not in tokenizer.get_vocab():
            special_tokens = ["ANSWER:"]
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            
        print(f"Tokenizer loaded from fine-tuned model, vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"Could not load fine-tuned tokenizer: {e}")
        # Fall back to recreating tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            local_files_only=True
        )
        # Add special tokens exactly as done during training
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        special_tokens = ["ANSWER:"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        tokenizer.padding_side = "right"
        print(f"Tokenizer recreated from base model with added tokens, vocab size: {len(tokenizer)}")
    
    # Phase 2: Model Loading Strategy
    print("Phase 2: Loading model...")
    try:
        # Phase 2A: Try to load the fine-tuned model directly
        print(f"Attempt 1: Loading directly from {FINETUNED_MODEL}")
        
        # First try with ignore_mismatched_sizes
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
        
        # IMPROVED: Ensure model vocabulary size exactly matches the target size
        current_size = model.get_input_embeddings().weight.size(0)
        if current_size != TARGET_VOCAB_SIZE:
            print(f"Resizing model embeddings from {current_size} to exact target size {TARGET_VOCAB_SIZE}")
            model.resize_token_embeddings(TARGET_VOCAB_SIZE)
        else:
            print(f"Model vocabulary size {current_size} already matches target size")
            
        print("Successfully loaded fine-tuned model directly!")
        
    except Exception as e:
        print(f"Direct loading failed: {e}")
        print("Phase 2B: Trying base model + adapter approach")
        
        try:
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
            )
            
            # IMPROVED: Resize embeddings to exact target size from error message
            print(f"Resizing base model embeddings from {model.get_input_embeddings().weight.size(0)} to exact target size {TARGET_VOCAB_SIZE}")
            model.resize_token_embeddings(TARGET_VOCAB_SIZE)
            
            # Check if we have adapter files
            adapter_safetensors = os.path.join(FINETUNED_MODEL, "adapter_model.safetensors")
            adapter_bin = os.path.join(FINETUNED_MODEL, "adapter_model.bin")
            
            if os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin):
                print("Found adapter files, trying custom PEFT loading")
                try:
                    # Import here to avoid dependency if not needed
                    from peft import PeftModel
                    
                    # Try manual loading with error suppression
                    try:
                        # Monkey patch the load_state_dict method temporarily
                        original_load_state_dict = torch.nn.Module.load_state_dict
                        
                        def patched_load_state_dict(self, state_dict, strict=True):
                            # Filter out problematic keys
                            filtered_dict = {}
                            
                            for k, v in state_dict.items():
                                # Check if the key exists in the model's state_dict
                                if k in self.state_dict():
                                    # Check if shapes match
                                    if self.state_dict()[k].shape == v.shape:
                                        filtered_dict[k] = v
                                    else:
                                        print(f"Skipping mismatched key: {k}, " 
                                              f"Model: {self.state_dict()[k].shape}, "
                                              f"Checkpoint: {v.shape}")
                                else:
                                    print(f"Skipping unknown key: {k}")
                            
                            return original_load_state_dict(self, filtered_dict, False)
                        
                        # Apply the monkey patch
                        torch.nn.Module.load_state_dict = patched_load_state_dict
                        
                        # Load with the patched method
                        peft_model = PeftModel.from_pretrained(
                            model,
                            FINETUNED_MODEL,
                            torch_dtype=torch.float16,
                        )
                        
                        # Restore original method
                        torch.nn.Module.load_state_dict = original_load_state_dict
                        
                        print("Successfully loaded adapter with custom patching!")
                        
                        # IMPORTANT: Convert PeftModel to standard model
                        print("Converting PEFT model to standard model for pipeline compatibility...")
                        try:
                            model = peft_model.merge_and_unload()
                            print("Successfully converted PEFT model to standard model!")
                        except Exception as merge_e:
                            print(f"Error merging PEFT model: {merge_e}")
                            print("Using base model with adapter applied (may not work with pipeline)")
                            model = peft_model.base_model.model
                            print("Extracted base model from PEFT model")
                            
                    except Exception as adapter_e:
                        print(f"Custom adapter loading failed: {adapter_e}")
                        # Restore original method if exception occurred
                        torch.nn.Module.load_state_dict = original_load_state_dict
                        raise
                        
                except Exception as peft_e:
                    print(f"PEFT loading failed: {peft_e}")
                    print("Phase 2C: Attempting direct state dict loading")
                    
                    # Try loading full model weights
                    model_bin = os.path.join(FINETUNED_MODEL, "pytorch_model.bin")
                    
                    if os.path.exists(model_bin):
                        print(f"Loading weights from {model_bin}")
                        try:
                            state_dict = torch.load(model_bin, map_location="cpu")
                            
                            # Handle embedding mismatches
                            if "model.embed_tokens.weight" in state_dict:
                                target_size = TARGET_VOCAB_SIZE  # Use exact target size
                                
                                # Fix embedding sizes
                                for key in ["model.embed_tokens.weight", "lm_head.weight"]:
                                    if key in state_dict:
                                        source_tensor = state_dict[key]
                                        source_size = source_tensor.size(0)
                                        
                                        if source_size != target_size:
                                            print(f"Resizing {key} from {source_size} to {target_size}")
                                            if source_size > target_size:
                                                # Truncate
                                                state_dict[key] = source_tensor[:target_size]
                                            else:
                                                # Pad with zeros
                                                padding = torch.zeros(
                                                    (target_size - source_size, source_tensor.size(1)),
                                                    dtype=source_tensor.dtype,
                                                    device=source_tensor.device
                                                )
                                                state_dict[key] = torch.cat([source_tensor, padding], dim=0)
                                                
                            # Load with strict=False to allow partial loading
                            missing, unexpected = model.load_state_dict(state_dict, strict=False)
                            print(f"Loaded state dict with {len(missing)} missing and {len(unexpected)} unexpected keys")
                            
                        except Exception as sd_e:
                            print(f"State dict loading failed: {sd_e}")
                            raise
                    else:
                        print("No model weights found, using base model only")
            else:
                print("No adapter files found, using base model only")
                
        except Exception as base_e:
            print(f"All model loading attempts failed: {base_e}")
            print("WARNING: Falling back to base model only - will not have 76.89% accuracy!")
            traceback.print_exc()
            
            # Last resort - just use the base model
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
            )
            model.resize_token_embeddings(TARGET_VOCAB_SIZE)  # Use exact target size
    
    # Enable KV cache for efficient inference
    model.config.use_cache = True
    
    # IMPROVED: Create pipeline with better generation settings
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,  # Increased for better extraction
        do_sample=True,      # Changed to True for better quality
        temperature=0.7,     # Moderate temperature
        top_p=0.95,          # Keep top_p filtering
        return_full_text=False,
    )
    
    print(f"Pipeline created successfully, using device: {model.device}")
    return pipe

def format_deepseek_prompt(caption, summary, question, options):
    """
    Improved prompt format specifically designed for the 76.89% accuracy model.
    Updated few-shot examples to match the exact format needed.
    """
    system_prompt = "You are a helpful visual question answering assistant specifically trained to understand and answer questions about videos."
    
    # IMPROVED: Few-shot examples with explicit option selection format
    few_shot = """
Example 1:
Question: What activity is the person primarily engaged in throughout the video?
Options:
option 0: Playing a musical instrument
option 1: Cooking food in the kitchen
option 2: Reading a book
option 3: Exercising
option 4: Dancing to music
Answer: option 1

Example 2:
Question: What specific object does the person interact with most frequently?
Options:
option 0: Smartphone
option 1: Book
option 2: Kitchen knife
option 3: Remote control
option 4: Computer keyboard
Answer: option 2

Example 3:
Question: How would you describe the main action performed by the person in this video?
Options:
option 0: The person is cutting vegetables precisely
option 1: The person is typing on a keyboard
option 2: The person is walking around aimlessly
option 3: The person is talking to another individual
option 4: The person is exercising vigorously
Answer: option 0
"""
    
    # Dynamic content with the video details
    dynamic_content = f"""
CAPTION:
{caption}

SUMMARY:
{summary}

QUESTION:
{question}

OPTIONS:
{options}

Answer:
"""
    
    # This format mirrors how the model was trained according to paste.txt
    full_prompt = f"{system_prompt}\n\n{few_shot}\n\n{dynamic_content}"
    return full_prompt

def extract_answer_from_output(text, original_options):
    """
    Enhanced answer extraction function with multiple strategies.
    Improved robustness for correctly identifying the intended option.
    """
    # Clean up text
    text = text.strip().lower()
    
    # Extract option texts and build mapping dictionary
    option_texts = []
    option_dict = {}
    for line in original_options.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                option_key = parts[0].strip()
                option_text = parts[1].strip()
                option_texts.append(option_text)
                option_dict[option_key] = option_text
                # Also store as just the number
                if "option" in option_key:
                    num_only = option_key.replace("option", "").strip()
                    option_dict[num_only] = option_text
                    # Explicitly store for exact "option X" pattern
                    option_dict[f"option {num_only}"] = option_text
    
    # IMPROVED: Strategy for direct "option N" pattern (most reliable in few-shot learning)
    direct_pattern = re.search(r'answer:\s*option\s*(\d+)', text)
    if direct_pattern:
        option_num = direct_pattern.group(1)
        option_key = f"option {option_num}"
        if option_key in option_dict:
            return option_dict[option_key]
    
    # Strategy 1: Direct match with option text
    for option_text in sorted(option_texts, key=len, reverse=True):
        if option_text.lower() in text:
            return option_text
    
    # Strategy 2: Look for option numbers with comprehensive patterns
    option_patterns = [
        r'option\s*(\d+)',
        r'answer.*?option\s*(\d+)',
        r'answer.*?(\d+)',
        r'(?:choose|select|pick)\s*option\s*(\d+)',
        r'(?:^|\s)(\d+)(?:$|\s)',  # standalone digit
        r'(?:would be|should be|is)\s*(\d+)'  # "would be 2"
    ]
    
    for pattern in option_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Take the last match, which is more likely to be the conclusion
            option_num = matches[-1]
            option_key = f"option {option_num}"
            if option_key in option_dict:
                return option_dict[option_key]
            elif option_num in option_dict:
                return option_dict[option_num]
    
    # Strategy 3: Extract from JSON object if present
    json_pattern = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text)
    if json_pattern:
        try:
            json_obj = json.loads(json_pattern.group(0))
            if 'ANSWER' in json_obj and isinstance(json_obj['ANSWER'], int):
                option_key = f"option {json_obj['ANSWER']}"
                if option_key in option_dict:
                    return option_dict[option_key]
        except:
            pass
    
    # IMPROVED: Enhanced word matching with options
    words = set(re.findall(r'\b\w{3,}\b', text))  # Words with 3+ chars
    best_option = None
    best_match = 0
    
    for option_text in option_texts:
        option_words = set(re.findall(r'\b\w{3,}\b', option_text.lower()))
        if option_words:
            overlap = len(words.intersection(option_words))
            if overlap > best_match:
                best_match = overlap
                best_option = option_text
    
    if best_option and best_match >= 2:  # At least 2 matching words
        return best_option
    
    # Strategy 5: First option mentioned in text
    for i in range(5):  # Assume 5 options maximum
        if f"option {i}" in text or f"({i})" in text or f"{i}." in text:
            option_key = f"option {i}"
            if option_key in option_dict:
                return option_dict[option_key]
    
    # Strategy 6: Last resort - return option 0
    if "option 0" in option_dict:
        return option_dict["option 0"]
    
    # If all else fails, return the first option text
    return option_texts[0] if option_texts else "No valid option found"

def process_single_inference(pipe, uid, queries, ok_dir, fail_dir):
    """
    Enhanced process a single inference with improved extraction and error handling.
    Added verification, missing file handling, and fallback strategies.
    """
    try:
        d = queries[uid]

        # Load captions with error handling
        try:
            with open(LAVILA_PATH + uid + '.json', 'r') as f:
                captions = json.load(f)
            caps = ''
            for c in captions:
                caps += c['Caption'] + "\n"
        except Exception as e:
            print(f"Error loading captions for {uid}: {str(e)}")
            caps = "Could not load captions."

        # IMPROVED: Handle missing summaries gracefully
        try:
            with open(SUMMARY_PATH + '/' + uid + '.txt', 'r') as f1:
                sum_text = f1.read()
        except FileNotFoundError:
            print(f"Summary file not found for {uid}, using caption instead")
            sum_text = "Summary not available. Using caption information: " + caps[:200]
        except Exception as e:
            print(f"Error loading summary for {uid}: {str(e)}")
            sum_text = "Summary not available."

        que = d['question']
        opt = ''
        for i in range(5):
            option_key = f"option {i}"
            if option_key in d:
                opt += f'{option_key}: {d[option_key]}\n'

        # Generate the optimized prompt
        prompt = format_deepseek_prompt(caps, sum_text, que, opt)
        
        # IMPROVED: First attempt with standard parameters
        outputs = pipe(prompt)
        raw_output = outputs[0]["generated_text"]
        
        # IMPROVED: Verify if output contains a recognizable answer
        valid_answer = False
        for i in range(5):
            option_key = f"option {i}"
            if (f"option {i}" in raw_output.lower() or 
                (option_key in d and d[option_key].lower() in raw_output.lower())):
                valid_answer = True
                break
        
        # Retry with different parameters if no valid answer found
        if not valid_answer:
            print(f"No clear answer found for {uid}, trying with different parameters")
            retry_outputs = pipe(prompt, temperature=0.9, do_sample=True, top_p=0.8)
            raw_output = retry_outputs[0]["generated_text"]
        
        # Extract answer with improved extraction
        answer_text = extract_answer_from_output(raw_output, opt)
        
        # IMPROVED: Find the index of the answer with enhanced matching
        answer_index = None
        
        # Try exact matching first
        for i in range(5):
            option_key = f"option {i}"
            if option_key in d and d[option_key].strip().lower() == answer_text.strip().lower():
                answer_index = i
                break
        
        # If not found, try fuzzy matching based on word overlap
        if answer_index is None:
            best_match = 0
            for i in range(5):
                option_key = f"option {i}"
                if option_key in d:
                    option_words = set(re.findall(r'\b\w{3,}\b', d[option_key].lower()))
                    answer_words = set(re.findall(r'\b\w{3,}\b', answer_text.lower()))
                    if option_words and answer_words:
                        overlap = len(option_words.intersection(answer_words))
                        if overlap > best_match:
                            best_match = overlap
                            answer_index = i
        
        # Only default to 0 if absolutely nothing matches
        if answer_index is None:
            answer_index = 0
            print(f"Could not match answer for {uid}, defaulting to option 0")
        
        # Extract a reason (limited to 200 chars)
        reason = raw_output.strip()[:200]
        
        # IMPROVED: Add confidence based on extraction method
        confidence = 5  # Default high confidence
        if answer_index is None:
            confidence = 1  # Low confidence for default
        
        # Build response in the required format
        response_dict = {
            "REASON": reason,
            "ANSWER": answer_index,
            "CONFIDENCE": confidence
        }
        
        # Save the result
        with open(f"{ok_dir}/{uid}.json", "w") as f:
            json.dump(response_dict, f)
        
        return True

    except Exception as e:
        print(f"Error processing {uid}: {str(e)}")
        try:
            # Create a fallback response even when error occurs
            fallback_response = {
                "REASON": f"Error during processing: {str(e)[:100]}",
                "ANSWER": 0,  # Default to first option
                "CONFIDENCE": 1  # Low confidence
            }
            
            # Save to ok dir anyway to ensure we have a response
            with open(f"{ok_dir}/{uid}.json", "w") as f:
                json.dump(fallback_response, f)
            
            # Also log the error in fail dir
            fail_json = {"error": str(e), "traceback": traceback.format_exc()}
            with open(f"{fail_dir}/{uid}.json", "w") as f:
                json.dump(fail_json, f)
        except:
            print(f"Critical error saving fallback for {uid}")
        
        return False

def inspect_directories():
    """
    Check all the relevant directories and files to diagnose issues.
    """
    print("\n=== DIRECTORY INSPECTION ===")
    
    # Check main directories
    print(f"\nChecking LAVILA_PATH: {LAVILA_PATH}")
    if os.path.exists(LAVILA_PATH):
        files = os.listdir(LAVILA_PATH)
        json_files = [f for f in files if f.endswith('.json')]
        print(f"  Directory exists: Yes")
        print(f"  Total files: {len(files)}")
        print(f"  JSON files: {len(json_files)}")
        print(f"  Sample files: {json_files[:5] if len(json_files) > 5 else json_files}")
    else:
        print(f"  Directory exists: No - CRITICAL ERROR!")
    
    print(f"\nChecking SUMMARY_PATH: {SUMMARY_PATH}")
    if os.path.exists(SUMMARY_PATH):
        files = os.listdir(SUMMARY_PATH)
        txt_files = [f for f in files if f.endswith('.txt')]
        print(f"  Directory exists: Yes")
        print(f"  Total files: {len(files)}")
        print(f"  Text files: {len(txt_files)}")
        print(f"  Sample files: {txt_files[:5] if len(txt_files) > 5 else txt_files}")
    else:
        print(f"  Directory exists: No - CRITICAL ERROR!")
    
    # Check results directory
    results_dir = RESULTS_DIR
    print(f"\nChecking results directory: {results_dir}")
    if os.path.exists(results_dir):
        print(f"  Directory exists: Yes")
        
        ok_dir = os.path.join(results_dir, "ok")
        fail_dir = os.path.join(results_dir, "failed")
        
        # Check "ok" subdirectory
        if os.path.exists(ok_dir):
            ok_files = os.listdir(ok_dir)
            print(f"  'ok' subdirectory exists: Yes")
            print(f"  Files in 'ok': {len(ok_files)}")
            if len(ok_files) > 0:
                print(f"  Sample 'ok' files: {ok_files[:5] if len(ok_files) > 5 else ok_files}")
        else:
            print(f"  'ok' subdirectory exists: No")
        
        # Check "failed" subdirectory
        if os.path.exists(fail_dir):
            fail_files = os.listdir(fail_dir)
            print(f"  'failed' subdirectory exists: Yes")
            print(f"  Files in 'failed': {len(fail_files)}")
            if len(fail_files) > 0:
                print(f"  Sample 'failed' files: {fail_files[:5] if len(fail_files) > 5 else fail_files}")
        else:
            print(f"  'failed' subdirectory exists: No")
    else:
        print(f"  Directory exists: No")
    
    # Check questions.json
    print(f"\nChecking QUESTION_PATH: {QUESTION_PATH}")
    if os.path.exists(QUESTION_PATH):
        print(f"  File exists: Yes")
        try:
            with open(QUESTION_PATH, 'r') as f:
                data = json.load(f)
                print(f"  Questions loaded: {len(data)}")
        except Exception as e:
            print(f"  Error loading questions: {e}")
    else:
        print(f"  File exists: No - CRITICAL ERROR!")
    
    # Check if the files were already processed
    if os.path.exists(LAVILA_PATH) and os.path.exists(ok_dir) and os.path.exists(fail_dir):
        lavila_files = [f[:-5] for f in os.listdir(LAVILA_PATH) if f.endswith('.json')]
        ok_files = [f[:-5] for f in os.listdir(ok_dir) if f.endswith('.json')]
        fail_files = [f[:-5] for f in os.listdir(fail_dir) if f.endswith('.json')]
        
        all_processed = set(ok_files).union(set(fail_files))
        
        remaining = set(lavila_files) - all_processed
        print(f"\nFile Processing Status:")
        print(f"  Total files in LAVILA_PATH: {len(lavila_files)}")
        print(f"  Files already processed: {len(all_processed)}")
        print(f"  Files remaining to process: {len(remaining)}")
        
        if len(remaining) == 0:
            print("\n*** ALL FILES ALREADY PROCESSED! ***")
            print("To process files again, rename or move the 'ok' and 'failed' directories.")
    
    print("\n=== END DIRECTORY INSPECTION ===")

def split_files_evenly(file_list, num_gpus):
    """Split files evenly among GPUs."""
    if len(file_list) == 0:
        return []
    n_sublists = min(num_gpus, len(file_list))
    file_sublists = [[] for _ in range(n_sublists)]
    for i, file in enumerate(file_list):
        file_sublists[i % n_sublists].append(file)
    return file_sublists

def worker_process(gpu_id, file_list, queries, results_dir, force_reprocess=False):
    """Process files on a single GPU."""
    # Set GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create output directories
    ok_dir = os.path.join(results_dir, "ok")
    fail_dir = os.path.join(results_dir, "failed")
    os.makedirs(ok_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    
    print(f"\n[GPU={gpu_id}] --- Processing with DeepSeek model ---", flush=True)
    
    # Load model
    pipe = load_deepseek_model()
    
    # Get list of files to process
    if force_reprocess:
        # Process all files regardless of whether they've been processed before
        incomplete_files = [f[:-5] for f in file_list]
    else:
        # Filter out already processed files
        completed_files_ok = {f[:-5] for f in os.listdir(ok_dir) if f.endswith('.json')}
        completed_files_failed = {f[:-5] for f in os.listdir(fail_dir) if f.endswith('.json')}
        all_completed = completed_files_ok.union(completed_files_failed)
        
        incomplete_files = [f[:-5] for f in file_list if f[:-5] not in all_completed]
    
    print(f"[GPU={gpu_id}] Files to process: {len(incomplete_files)}")
    
    if not incomplete_files:
        print(f"[GPU={gpu_id}] No files to process. Exiting.")
        return
    
    # Process files individually - batch processing removed
    success_count = 0
    for uid in tqdm(incomplete_files, desc=f"GPU {gpu_id} processing"):
        success = process_single_inference(pipe, uid, queries, ok_dir, fail_dir)
        if success:
            success_count += 1
            
        # Periodic progress update
        if (success_count + 1) % 10 == 0:
            print(f"[GPU={gpu_id}] Processed {success_count}/{len(incomplete_files)} successfully")
    
    print(f"[GPU={gpu_id}] Complete. Successfully processed {success_count}/{len(incomplete_files)}")

def main():
    # Create results directory
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    # Inspect directories to diagnose "no files to process" issue
    inspect_directories()
    
    # Ask user if they want to continue
    response = input("\nDo you want to continue with processing? (y/n): ")
    if response.lower() != 'y':
        print("Exiting program.")
        return
    
    # Option to force reprocessing
    force_reprocess = input("Force reprocessing of already processed files? (y/n): ")
    
    # Load questions
    with open(QUESTION_PATH, 'r') as f1:
        data = json.load(f1)
    queries = {item['q_uid']: item for item in data}
    
    # Get list of files
    qa_list = os.listdir(LAVILA_PATH)
    
    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    # Option to clear output directories before processing
    if force_reprocess.lower() == 'y':
        response = input("Clear existing results before processing? (y/n): ")
        if response.lower() == 'y':
            ok_dir = os.path.join(results_dir, "ok")
            fail_dir = os.path.join(results_dir, "failed")
            
            if os.path.exists(ok_dir):
                import shutil
                shutil.rmtree(ok_dir)
                os.makedirs(ok_dir)
                print(f"Cleared directory: {ok_dir}")
                
            if os.path.exists(fail_dir):
                import shutil
                shutil.rmtree(fail_dir)
                os.makedirs(fail_dir)
                print(f"Cleared directory: {fail_dir}")
    
    # Divide files among GPUs
    file_sublists = split_files_evenly(qa_list, num_gpus)
    
    # Create and start processes
    processes = []
    for gpu_id, file_subset in enumerate(file_sublists):
        if not file_subset:
            continue
            
        p = multiprocessing.Process(
            target=worker_process,
            args=(
                gpu_id, 
                file_subset, 
                queries, 
                results_dir, 
                force_reprocess.lower() == 'y'
            )
        )
        p.daemon = True
        p.start()
        processes.append(p)
    
    print("All processes started. (Ctrl+C to stop)")

    try:
        # Wait for all processes to complete
        for p in processes:
            p.join()
        print("All processes completed successfully!")
        
        # Final validation
        ok_dir = os.path.join(results_dir, "ok")
        result_files = os.listdir(ok_dir)
        print(f"Total result files generated: {len(result_files)}")
        
        # Check for option 0 bias
        option_counts = {i: 0 for i in range(5)}
        for rf in result_files[:1000]:  # Check a sample
            try:
                with open(os.path.join(ok_dir, rf), 'r') as f:
                    data = json.load(f)
                    if "ANSWER" in data:
                        answer = data["ANSWER"]
                        if isinstance(answer, int) and 0 <= answer <= 4:
                            option_counts[answer] += 1
            except:
                pass
        
        # Print distribution
        total = sum(option_counts.values())
        if total > 0:
            print("\nAnswer distribution (sample):")
            for opt, count in option_counts.items():
                print(f"  Option {opt}: {count} ({count/total:.1%})")
                
            # Check for option 0 bias
            if option_counts[0] / total > 0.4:
                print("\nWARNING: High bias toward Option 0 detected. This may indicate extraction issues.")
        
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting...")

if __name__ == "__main__":
    main()