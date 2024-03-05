import argparse
import json
import os

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from datasets import process_data

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str,
                    default="../saved_models/peft_alpaca_en_llama2-chat-7b-merged-best")
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False,
                    default='results/peft/catqa')
parser.add_argument('--save_name', help='result save names', type=str,
                    default='reference_english.json')
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int,
                    default=-1)
parser.add_argument('--load_8bit', help='for open source models-if the model to be loaded in 8 bit', action='store_true', required=False)
parser.add_argument('--dataset_path', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', type=str,
                    default='harmful_questions/catqa/catqa_english.json')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sft_response', help='', action='store_true')

args = parser.parse_args()

dataset_path = args.dataset_path
model_name = args.model_path
sft_response = args.sft_response
save_path = args.save_path
save_name = args.save_name
load_in_8bit = args.load_8bit
num_samples = args.num_samples
seed = args.seed

if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"\n\nconfiguration")
print(f"*{'-'*10}*")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print(f"*{'-'*10}*\n\n")

# setting up model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token


if load_in_8bit:
    print("\n\n***loading model in 8 bits***\n\n")
else:
    # model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model = LLM(model=model_name, tokenizer=model_name, dtype='float16')

orig_que, prompt_que = process_data(dataset_path,
                                    sft_response=sft_response,
                                    tokenizer=tokenizer,
                                    num_samples=num_samples,
                                    seed=seed)

# generate responses setting up
outputs = []
system_message = ''

print("\ngenerating responses...")
batch_size = 8
for i in tqdm(range(0, len(prompt_que), batch_size)):

    inputs = prompt_que[i: i+batch_size]

    # inputs = tokenizer([inputs], return_tensors="pt", truncation=True, padding=True).to("cuda")
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=500)
    batch_outputs = model.generate(inputs, sampling_params, use_tqdm=False)

    for j, output in enumerate(batch_outputs):
        response = output.outputs[0].text
        question = orig_que[i+j]

        response = {'prompt': question, 'response': response}

        outputs.append(response)

with open(f'{save_path}/{save_name}', 'w', encoding='utf-8') as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)

print("***** Save inference results *****")
print("Sucessful save predictions to {}".format(f'{save_path}/{save_name}'))


'''
How to run?
    closed source:
        python generate_responses.py --model 'chatgpt' --prompt 'red_prompts/cou.txt' --dataset harmful_questions/dangerousqa.json --num_samples 10
        python generate_responses.py --model 'chatgpt' --prompt 'red_prompts/cou.txt' --dataset harmful_questions/dangerousqa.json --num_samples 10 --clean_thoughts

        python generate_responses.py --model 'gpt4' --prompt 'red_prompts/cou.txt' --dataset harmful_questions/dangerousqa.json --num_samples 10
        python generate_responses.py --model 'gpt4' --prompt 'red_prompts/cou.txt' --dataset harmful_questions/dangerousqa.json --num_samples 10 --clean_thoughts

        python generate_responses.py --model 'claude-instant-1' --prompt 'red_prompts/cotclaude.txt' --dataset harmful_questions/dangerousqa.json
        python generate_responses.py --model "claude-1.3" --prompt 'red_prompts/cotclaude.txt' --dataset harmful_questions/dangerousqa.json
        python generate_responses.py --model "claude-2" --prompt 'red_prompts/cotclaude.txt' --dataset harmful_questions/dangerousqa.json

    open source models:
        python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt 'red_prompts/cou.txt' --dataset harmful_questions/dangerousqa.json --num_samples 10
        python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt 'red_prompts/cou.txt' --dataset harmful_questions/dangerousqa.json --num_samples 10 --clean_thoughts

        python generate_responses.py --model meta-llama/Llama-2-7b-chat-hf --prompt 'red_prompts/standardllama2.txt' --dataset harmful_questions/adabenchqs.json --num_samples 10
        python generate_responses.py --model /data/rishabh/flacuna/unaligned_llama2chat_7b_3_epoch_1960_samples --prompt 'red_prompts/standardllama2.txt' --dataset harmful_questions/adabenchqs.json --num_samples 10
'''
