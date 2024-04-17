import argparse
from typing import List
import torch
from torch import nn
import os
import sys
print(sys.path)
from tasks.task_arithmetic import task_arithmetic_func
from tasks.dare import dare_func
from tasks.ties_merging import ties_merging_func
from tasks.arithmetic import get_task_wise_weights
import shutil
from safetensors.torch import load_file

parser = argparse.ArgumentParser(description='Evaluate an arithmetic expression')
parser.add_argument('--adapter_name_or_paths',
                    nargs='+',
                    default=['/media/data/2/yx/model_merging/saved_models/sft/Safe-WizardLM-7b-sft-alpaca_zh/checkpoint-1500',
                             '/media/data/2/yx/model_merging/saved_models/sft/Safe-WizardLM-7b-sft-alpaca_en/checkpoint-1500',
                             '/media/data/2/yx/model_merging/saved_models/sft/Safe-WizardLM-7b-sft-code/checkpoint-900',
                             '/media/data/2/yx/model_merging/saved_models/sft/Safe-WizardLM-7b-sft-hindi/checkpoint-2400',
                             '/media/data/2/yx/model_merging/saved_models/sft/Safe-WizardLM-7b-sft-math/checkpoint-300'],
                    help='Path to the adapter weight or identifier from huggingface.co/models.')

parser.add_argument('--base_adapter_name_or_path',
                    type=str,
                    default='/media/data/2/yx/model_merging/saved_models/pretrain/Safe-WizardLM-7b-pretrain_sft_after_dpo/checkpoint-4371',
                    help='Path to the adapter weight or identifier from huggingface.co/models.')
parser.add_argument('--output_dir',
                    type=str,
                    default='/media/data/2/yx/model_merging/saved_models/multi_sft/dare',
                    help='The output directory to save the merged adapter.')

parser.add_argument('--gamma',
                    type=float,
                    default=0.5,
                    help='safety vector coefficient.')

parser.add_argument('--unsafe_vector_path',
                    type=str,
                    default=None,
                    help='The path to the safe vector.')

args = parser.parse_args()
# 输出参数及其值
for arg, value in args.__dict__.items():
    print(arg, ':', value)

base_adapter_name_or_path = args.base_adapter_name_or_path
adapter_name_or_paths = args.adapter_name_or_paths
unsafe_vector_path = args.unsafe_vector_path
output_dir = args.output_dir


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义一个函数用于将bf16转换为float16
def to_float16(value):
    if value.dtype == torch.bfloat16:
        return torch.tensor(value, dtype=torch.float16)
    elif value.dtype == torch.float32:
        return value.half()
    else:
        return value


if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
task_vectors = []
if 'adapter_model.bin' in os.listdir(base_adapter_name_or_path):
    base_adapter_weight = torch.load(os.path.join(base_adapter_name_or_path, 'adapter_model.bin'), map_location="cpu")
    source_path = os.path.join(base_adapter_name_or_path, 'adapter_config.json')  # 源文件路径
    shutil.copy(source_path, output_dir)
    saved_adapter_weight_path = os.path.join(output_dir, 'adapter_model.bin')
    unsafe_vector_weight = None
else:
    if 'pytorch_model.bin' in os.listdir(base_adapter_name_or_path):
        base_adapter_weight = torch.load(os.path.join(base_adapter_name_or_path, 'pytorch_model.bin'), map_location="cpu")
        unsafe_vector_weight = load_file(os.path.join(unsafe_vector_path, 'model.safetensors'))
    # 遍历字典中的值，并将bf16转换为float16
    for name, value in base_adapter_weight.items():
        base_adapter_weight[name] = to_float16(value)

    for name, value in unsafe_vector_weight.items():
        unsafe_vector_weight[name] = to_float16(value)

    source_path = os.path.join(base_adapter_name_or_path, 'config.json')  # 源文件路径
    shutil.copy(source_path, output_dir)
    source_path = os.path.join(base_adapter_name_or_path, 'special_tokens_map.json')
    shutil.copy(source_path, output_dir)
    source_path = os.path.join(base_adapter_name_or_path, 'tokenizer_config.json')
    shutil.copy(source_path, output_dir)
    source_path = os.path.join(base_adapter_name_or_path, 'tokenizer.model')
    shutil.copy(source_path, output_dir)
    saved_adapter_weight_path = os.path.join(output_dir, 'pytorch_model.bin')


for adapter_name_or_path in adapter_name_or_paths:
    if 'adapter_model.bin' in os.listdir(adapter_name_or_path):
        current_adapter_weight = torch.load(os.path.join(adapter_name_or_path, 'adapter_model.bin'), map_location="cpu")
    else:
        current_adapter_weight = torch.load(os.path.join(adapter_name_or_path, 'pytorch_model.bin'), map_location="cpu")

safe_task_vector = {}
if unsafe_vector_weight is None:
    for key in base_adapter_weight.keys():
        safe_task_vector[key] = base_adapter_weight[key].to(device)

else:
    for key in base_adapter_weight.keys():
        safe_task_vector[key] = (base_adapter_weight[key] - unsafe_vector_weight[key]).to(device)


for key in current_adapter_weight.keys():
    current_adapter_weight[key] = current_adapter_weight[key].to(device)


saved_adapter_weight = {}
for key, value in current_adapter_weight.items():
    saved_adapter_weight[key] = current_adapter_weight[key] + args.gamma * safe_task_vector[key]

torch.save(saved_adapter_weight, saved_adapter_weight_path)



