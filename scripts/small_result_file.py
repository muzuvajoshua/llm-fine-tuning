import json

# Step 1: Read the existing result.json file
with open('result.json', 'r') as f:
    full_results = json.load(f)

# Step 2: Extract the first 5 entries
small_results = {}
count = 0
for key, value in full_results.items():
    small_results[key] = value
    count += 1
    if count >= 5:
        break

# Step 3: Save to a new file
with open('small_result.json', 'w') as f:
    json.dump(small_results, f, indent=4)

print(f"Created small_result.json with {len(small_results)} entries")