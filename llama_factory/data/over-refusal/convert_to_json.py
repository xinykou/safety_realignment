import pandas as pd
import json

data = pd.read_csv('data/over-refusal/or-bench-hard-1k.csv')
output_file_name = 'data/over-refusal/or-bench.json'

all_categories = {}
max_num_per_category = 10
filtered_data = []

for i in range(len(data)):
    instance = {}
    prompt = data.iloc[i]['prompt']
    category = data.iloc[i]['category']
    if category not in all_categories:
        all_categories[category] = 0
        instance['prompt'] = prompt
        filtered_data.append(instance)
        all_categories[category] += 1

    elif all_categories[category] < max_num_per_category:
        instance['prompt'] = prompt
        filtered_data.append(instance)
        all_categories[category] += 1


with open(output_file_name, 'w') as f:
    json.dump(filtered_data, f, indent=4)

