import json
import random

import jsonlines as jsl


def process_data(dataset_path,
                 response_path=None,
                 tokenizer=None,
                 num_samples=-1,
                 sft_response=False,
                 seed=42):
    # category: 14, total: 700 --> select: 140
    # https://huggingface.co/datasets/PKU-Alignment/BeaverTails-Evaluation
    if 'BeaverTails' in dataset_path:
        category_samples = {}
        clip_category_samples = 10
        data = list(jsl.open(dataset_path))
        original_questions = []
        for d in data:
            if d['category'] not in category_samples:
                category_samples[d['category']] = []
            category_samples[d['category']].append(d)

        for category, samples in category_samples.items():
            clip_data = samples[:clip_category_samples]
            original_questions.extend([d['prompt'] for d in clip_data])

    # category: 5, sub_category: 11, total: 550. --> select:110
    # https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA
    elif 'catqa' in dataset_path:
        data = json.load(open(dataset_path))
        prompt_questions = []
        original_questions = []
        clip_sub_category_samples = 2
        for key, value in data.items():
            sub_data = value
            for sub_key, sub_value in sub_data.items():
                clip_data = sub_value[:clip_sub_category_samples]
                original_questions.extend(clip_data)

    # category: 10, sub_category: 9 or 10, total: 1960.-->select: 196
    # https://huggingface.co/datasets/declare-lab/HarmfulQA
    elif 'harmfulqa' in dataset_path:
        data = json.load(open(dataset_path))
        prompt_questions = []
        original_questions = []
        clip_sub_category_samples = 2
        for key, value in data.items():
            sub_data = value
            for sub_key, sub_value in sub_data.items():
                clip_data = sub_value[:clip_sub_category_samples]
                original_questions.extend(clip_data)

    # category: 10, total: 200. --->select: 200
    # https://huggingface.co/datasets/CherryDurian/shadow-alignment
    elif 'shadow-alignment' in dataset_path:
        data = json.load(open(dataset_path))
        original_questions = [d['prompt'] for d in data]

    # total : 200
    # https://github.com/SALT-NLP/chain-of-thought-bias/blob/main/data/dangerous-q/toxic_outs.json
    elif 'dangerousqa' in dataset_path:
        data = json.load(open(dataset_path))
        original_questions = data
    else:
        raise ValueError("Dataset not found")

    num_ = len(original_questions)

    if num_samples != -1 and num_samples < num_:
        random.seed(seed)
        indices = random.sample(range(num_), num_samples)
        original_questions = [original_questions[i] for i in indices]
    else:
        pass

    return original_questions





if __name__ == "__main__":
    # "BeaverTails"
    # data_path = "../harmful_questions/BeaverTails/test.jsonl"
    # out_path = "./BeaverTails.json"
    # "catqa"
    data_path = "../harmful_questions/catqa/catqa_english.json"

    # "harmfulqa"
    # data_path = "../harmful_questions/harmfulqa/harmfulqa.json"
    # out_path = "./harmfulqa.json"
    # "shadow-alignment"
    # data_path = "../harmful_questions/shadow-alignment/eval.json"
    # out_path = "./shadow-alignment.json"
    # "dangerousqa"
    # data_path = "../harmful_questions/dangerousqa/dangerousqa.json"
    # out_path = "./dangerousqa.json"
    attack_types = ["cot", "cou", "suffix"]
    for attack_type in attack_types:
        with open(f"./{attack_type}.txt", 'r') as f:
            attack_prompt = f.read()

        out_path = f"./catqa_english--{attack_type}.json"
        questions = process_data(data_path, seed=42)
        format_data = []
        for d in questions:
            inputs = attack_prompt.replace("<question>", d)
            format_data.append({"prompt": inputs})

        with open(out_path, 'w') as f:
            json.dump(format_data, f, indent=4)