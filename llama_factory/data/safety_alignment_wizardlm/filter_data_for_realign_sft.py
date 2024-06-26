"""""
filter response pairs (i.e. they are both unsafe or safe) for mask train
"""""

import json
import jsonlines as jsl

org_pretrain_path = "./pku-saferlhf-test.jsonl"
out_pretrain_path = "../pku_seferlhf_realign_sft_5k.json"

data = list(jsl.open(org_pretrain_path))
filtered_samples = []

for d in data:
    sample = {}
    if d["is_response_0_safe"] == d["is_response_1_safe"]:
        continue

    idx = d["safer_response_id"]
    if idx == 0:
        better_response = d["response_0"]
        worse_response = d["response_1"]
    elif idx == 1:
        better_response = d["response_1"]
        worse_response = d["response_0"]
    else:
        raise ValueError(f"idx: {idx}")

    sample["response"] = [better_response]

    sample["prompt"] = d["prompt"]

    filtered_samples.append(sample)

print()
with open(out_pretrain_path, 'w') as f:
    json.dump(filtered_samples, f, ensure_ascii=False, indent=4)






