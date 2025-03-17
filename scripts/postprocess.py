import os
import json
import re
import sys

def build_id_dict(folder_path):
    """Extract answer IDs from result JSONs in the specified folder"""
    id_dict = {}
    
    # Count for statistics
    processed = 0
    errors = 0
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"ERROR: Path does not exist: {folder_path}")
        sys.exit(1)
        
    print(f"Reading result files from: {folder_path}")
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            processed += 1
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract the answer value - should be an integer 0-4
                id_value = int(data["ANSWER"])
                
                # Validate range
                if id_value < 0 or id_value > 4:
                    print(f"Warning: {filename} has invalid ANSWER value: {id_value}")
                
                # Get the file ID without extension
                file_id = filename.split('.')[0]
                
                # Store both answer and confidence
                id_dict[file_id] = [id_value, data.get("CONFIDENCE", 5)]
                
            except Exception as e:
                errors += 1
                print(f"Error processing {filename}: {str(e)}")
            
            # Show progress every 1000 files
            if processed % 1000 == 0:
                print(f"Processed {processed} files so far...")
    
    print(f"Processed {processed} files with {errors} errors")
    return id_dict

def main():
    # Path to DeepSeek results
    # folder_path1 = '/mnt/st2/dir_ikeda/panasonic/HCQA/results/result_deepseek-tuned-76percent/ok'
    folder_path = '/mnt/st1/dir_ikeda/panasonic/HCQA/results/result_deepseek-base-only/ok'
    
    # Process results
    result_dict = build_id_dict(folder_path)
    print(f"Total entries in result dictionary: {len(result_dict)}")

    # Create final answer dictionary (just IDs)
    data = {}
    for key, value in result_dict.items():
        data[key] = int(value[0])

    # Check if subset_answers.json exists and incorporate if it does
    subset_path = '/mnt/st2/dir_ikeda/panasonic/HCQA/subset_answers.json'
    if os.path.exists(subset_path):
        print(f"Loading additional answers from {subset_path}")
        
        # Add this line to track before count
        before_count = len(data)
        
        subset = json.load(open(subset_path))
        for key, value in subset.items():
            data[key] = int(value)
        
        # Add these lines to track after count
        after_count = len(data)
        added = after_count - before_count
        print(f"Added {added} entries from subset file")
        print(f"Total entries after adding subset: {after_count}")
        
        if added == 0:
            print("WARNING: No new entries were added from the subset file!")
            print("This might cause validation to fail.")
            
    else:
        print(f"WARNING: Subset file not found at {subset_path}")
        print("The validation will likely fail without the subset data!")
    
    # Check if we have a reasonable number of entries
    if len(data) < 4000:
        print(f"WARNING: Final result has only {len(data)} entries, which may be insufficient!")
        print("The validation may expect a larger dataset including the subset.")
    
    # Save results
    output_path = 'result3.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to {output_path}")
    print(f"Final file contains {len(data)} answers")

if __name__ == "__main__":
    main()