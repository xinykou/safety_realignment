import json
import random

import jsonlines as jsl


def process_data(dataset_path, topic=None):
    # category: 11, sub_category: 5, total: 550. --> select:110
    # https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA
    data = json.load(open(dataset_path))
    original_questions = []
    sub_data = data[topic]
    for sub_key, sub_value in sub_data.items():
        original_questions.extend(sub_value)

    return original_questions


if __name__ == "__main__":

    data_path = "../harmful_questions/catqa/catqa_english.json"
    topics = ["Illegal Activity",
              "Child Abuse",
              "Hate/Harass/Violence",
              "Malware Viruses",
              "Physical Harm",
              "Economic Harm",
              "Fraud/Deception",
              "Adult Content",
              "Political Campaigning",
              "Privacy Violation Activity",
              "Tailored Financial Advice"]
    for topic in topics:

        questions = process_data(data_path, topic=topic)
        format_data = []
        for d in questions:
            format_data.append({"prompt": d})

        if '/' in topic:
            topic = topic.replace('/', '_')
        elif ' ' in topic:
            topic = topic.replace(' ', '_')
        else:
            topic = topic

        out_path = f"./catqa_english-{topic}.json"
        with open(out_path, 'w') as f:
            json.dump(format_data, f, indent=4)