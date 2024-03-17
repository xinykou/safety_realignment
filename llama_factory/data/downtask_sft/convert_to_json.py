import pandas as pd
import json


# -----hindi------------------------------------------------
# 1. covert to json
file = "hindi_train-00000-of-00001-9c3cdcdad5ba72d6.parquet"

# Load the parquet file into a pandas DataFrame
df = pd.read_parquet(file)

# Save the DataFrame as a JSON file
df.to_json('./pre_train_hindi.json',
           orient='records', lines=False)

# 2. convert to "LLama Factory" data format
data = json.load(open('./pre_train_hindi.json', 'r'))
new_data = []
for i in range(len(data)):
    new_data.append({'instruction': data[i]['instruction'], 'input': data[i]['input'], 'output': data[i]['output']})

with open('../alpaca_cleaned_hindi.json', 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

# -----gsm8k------------------------------------------------
# 1. covert to json
file = "gsm8k_train-00000-of-00001.parquet"

# Load the parquet file into a pandas DataFrame
df = pd.read_parquet(file)

# Save the DataFrame as a JSON file
df.to_json('./pre_train_gsm8k.json',
           orient='records', lines=False)

# 2. convert to "LLama Factory" data format
data = json.load(open('./pre_train_gsm8k.json', 'r'))
new_data = []
for i in range(len(data)):
    new_data.append({'instruction': data[i]['question'], 'output': data[i]['answer']})

with open('../gsm8k.json', 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)


# -----openai humaneval------------------------------------------------
# 1. covert to json
# file = "humaneval-test-00000-of-00001.parquet"
#
# # Load the parquet file into a pandas DataFrame
# df = pd.read_parquet(file)
#
# # Save the DataFrame as a JSON file
# df.to_json('./test_humaneval.json',
#            orient='records', lines=False)
#
# # 2. convert to "LLama Factory" data format
# data = json.load(open('./pre_train_gsm8k.json', 'r'))
# new_data = []
# for i in range(len(data)):
#     new_data.append({'input': data[i]['question'], 'output': data[i]['answer']})
#
# with open('../openai_humaneval.json', 'w') as f:
#     json.dump(new_data, f, indent=4, ensure_ascii=False)
