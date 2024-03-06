import json
import jsonlines as jsl
import random

def gen_prompt(que, tokenizer=None):
    chat = [{"role": "user", "content": que}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False) + "Sure,"
    return prompt

# deal with different unsafe datasets
def process_data(dataset_path,
                 response_path=None,
                 tokenizer=None,
                 num_samples=-1,
                 sft_response=False,
                 seed=None):
    # category: 14, total: 700 --> select: 140
    # https://huggingface.co/datasets/PKU-Alignment/BeaverTails-Evaluation
    if 'BeaverTails' in dataset_path:
        category_samples = {}
        clip_category_samples = 10
        data = list(jsl.open(dataset_path))
        original_questions = []
        prompt_questions = []
        for d in data:
            if d['category'] not in category_samples:
                category_samples[d['category']] = []
            category_samples[d['category']].append(d)

        for category, samples in category_samples.items():
            clip_data = samples[:clip_category_samples]
            prompt_questions.extend([gen_prompt(d['prompt'], tokenizer=tokenizer) for d in clip_data])
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
                prompt_questions.extend(gen_prompt(d, tokenizer=tokenizer) for d in clip_data)
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
                prompt_questions.extend(gen_prompt(d, tokenizer=tokenizer) for d in clip_data)
                original_questions.extend(clip_data)

    # category: 10, total: 200. --->select: 200
    # https://huggingface.co/datasets/CherryDurian/shadow-alignment
    elif 'shadow-alignment' in dataset_path:
        data = json.load(open(dataset_path))
        original_questions = [d['prompt'] for d in data]
        prompt_questions = [gen_prompt(d['prompt'], tokenizer=tokenizer) for d in data]
    # total : 200
    # https://github.com/SALT-NLP/chain-of-thought-bias/blob/main/data/dangerous-q/toxic_outs.json
    elif 'dangerousqa' in dataset_path:
        data = json.load(open(dataset_path))
        original_questions = data
        prompt_questions = [gen_prompt(d, tokenizer=tokenizer) for d in data]
    else:
        raise ValueError("Dataset not found")

    num_ = len(prompt_questions)

    if num_samples != -1 and num_samples < num_:
        random.seed(seed)
        indices = random.sample(range(num_), num_samples)
        prompt_questions = [prompt_questions[i] for i in indices]
        original_questions = [original_questions[i] for i in indices]
    else:
        pass

    return original_questions, prompt_questions


def process_data_0(dataset_path,
                 response_path=None,
                 tokenizer=None,
                 num_samples=-1,
                 need_response=False,
                 seed=None):

    if 'BeaverTails' in dataset_path:
        data = list(jsl.open(dataset_path))

        prompt_questions = [gen_prompt(d['prompt'], tokenizer=tokenizer) for d in data]
        # prompt_questions = [d['prompt'] for d in data]
        original_questions = [d['prompt'] for d in data]
    elif 'catqa' in dataset_path:
        data = json.load(open(dataset_path))
        prompt_questions = []
        original_questions = []
        for key, value in data.items():
            sub_data = value
            for sub_key, sub_value in sub_data.items():
                conerted = [gen_prompt(d, tokenizer=tokenizer) for d in sub_value]
                prompt_questions.extend(conerted)
                original_questions.extend(sub_value)
    else:
        raise ValueError("Dataset not found")

    num_ = len(prompt_questions)

    if need_response:
        try:
            reference_data = json.load(open(response_path))
            reference_resp = [d['response'] for d in reference_data]
            num_ref = len(reference_resp)
            assert num_ref == num_  # check if the number of reference responses is equal to the number of questions
        except KeyError:
            raise KeyError(
                "response key not found, please firstly add reference responses by running the script")
    else:
        reference_resp = None

    if num_samples != -1:
        random.seed(seed)
        indices = random.sample(range(num_), num_samples)
        prompt_questions = [prompt_questions[i] for i in indices]
        original_questions = [original_questions[i] for i in indices]
        if need_response:
            reference_resp = [reference_resp[i] for i in indices]
    else:
        pass

    return original_questions, reference_resp, prompt_questions



