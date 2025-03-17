import os
import json
import argparse

def format_results(input_dir, output_file, subset_file=None):
    """
    Format results from the inference output directory for validation.
    
    Args:
        input_dir (str): Path to the "ok" directory containing result JSONs
        output_file (str): Path to save the formatted results
        subset_file (str, optional): Path to subset_answers.json
    """
    print(f"Formatting results from: {input_dir}")
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Extract answers from results
    result_dict = {}
    processed = 0
    errors = 0
    
    print(f"Processing files in: {input_dir}")
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            processed += 1
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract the answer value
                if "ANSWER" in data:
                    answer_value = int(data["ANSWER"])
                    
                    # Validate range
                    if answer_value < 0 or answer_value > 4:
                        print(f"Warning: {filename} has invalid ANSWER value: {answer_value}")
                    
                    # Get the file ID without extension
                    file_id = filename.split('.')[0]
                    
                    # Store the answer
                    result_dict[file_id] = answer_value
            except Exception as e:
                errors += 1
                print(f"Error processing {filename}: {str(e)}")
                
            # Show progress every 1000 files
            if processed % 1000 == 0:
                print(f"Processed {processed} files so far...")
    
    print(f"Processed {processed} files with {errors} errors")
    print(f"Total entries in result dictionary: {len(result_dict)}")
    
    # Add subset answers if specified
    if subset_file and os.path.exists(subset_file):
        print(f"Loading additional answers from {subset_file}")
        
        # Track before count
        before_count = len(result_dict)
        
        try:
            with open(subset_file, 'r') as f:
                subset_answers = json.load(f)
                
            for uid, answer in subset_answers.items():
                # Only add if not already present
                if uid not in result_dict:
                    result_dict[uid] = int(answer)
                    
            # Track after count
            after_count = len(result_dict)
            added = after_count - before_count
            
            print(f"Added {added} entries from subset file")
            print(f"Total entries after adding subset: {after_count}")
            
            if added == 0:
                print("WARNING: No new entries were added from the subset file!")
                print("This might cause validation to fail.")
        except Exception as e:
            print(f"Error loading subset file: {e}")
    
    # Check if we have a reasonable number of entries
    if len(result_dict) < 4000:
        print(f"WARNING: Final result has only {len(result_dict)} entries, which may be insufficient!")
        print("The validation may expect a larger dataset including the subset.")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    print(f"Results saved to {output_file}")
    print(f"Final file contains {len(result_dict)} answers")
    
    return result_dict

def main():
    parser = argparse.ArgumentParser(description="Format model results for validation")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to directory containing result JSON files")
    parser.add_argument("--output_file", type=str, default="formatted_results.json",
                        help="Path to save the formatted results")
    parser.add_argument("--subset_file", type=str, default="/mnt/st2/dir_ikeda/panasonic/HCQA/subset_answers.json",
                        help="Path to subset_answers.json")
    
    args = parser.parse_args()
    
    format_results(args.input_dir, args.output_file, args.subset_file)

if __name__ == "__main__":
    main()