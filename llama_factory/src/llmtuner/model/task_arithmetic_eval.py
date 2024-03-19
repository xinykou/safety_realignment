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


parser = argparse.ArgumentParser(description='Evaluate an arithmetic expression')
parser.add_argument('--task_vectors_merged_methods',
                    type=str,
                    default='task_arithmetic',
                    help='The methods to merge the task vectors: task_arithmetic, ties_merging or dare')

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
                    default='/media/data/2/yx/model_merging/saved_models/multi_sft/task_arithmetic',
                    help='The output directory to save the merged adapter.')

parser.add_argument('--task_wise_weight',
                    type=float,
                    default=None,
                    help='The weight for each task.')


args = parser.parse_args()


task_vectors_merged_methods = args.task_vectors_merged_methods
base_adapter_name_or_path = args.base_adapter_name_or_path
adapter_name_or_paths = args.adapter_name_or_paths
output_dir = args.output_dir
task_wise_weight_init = args.task_wise_weight

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

init_task_wise_weights = get_task_wise_weights(num_models=len(adapter_name_or_paths),
                                               init_values=task_wise_weight_init)

task_wise_weights_all = nn.Parameter(init_task_wise_weights, requires_grad=False)
task_wise_weights_all = task_wise_weights_all.to(device)

task_vectors = []
base_adapter_weight = torch.load(os.path.join(base_adapter_name_or_path, 'adapter_model.bin'), map_location="cpu")


for adapter_name_or_path in adapter_name_or_paths:
    current_adapter_weight = torch.load(os.path.join(adapter_name_or_path, 'adapter_model.bin'), map_location="cpu")
    current_task_vector = {}
    for key in current_adapter_weight.keys():
        current_task_vector[key] = (current_adapter_weight[key] - base_adapter_weight[key]).to(device)

    task_vectors.append(current_task_vector)

if task_vectors_merged_methods == 'task_arithmetic':
    merged_task_vector = task_arithmetic_func(task_vectors, task_wise_weights=task_wise_weights_all)

else:
    raise ValueError(f"task_vectors_merged_methods must be task_arithmetic, got {task_vectors_merged_methods}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for key in base_adapter_weight.keys():
    base_adapter_weight[key] = base_adapter_weight[key].to(device)

saved_adapter_weight_path = os.path.join(output_dir, 'adapter_model.bin')
saved_adapter_weight = {}
for key, value in merged_task_vector.items():
    saved_adapter_weight[key] = base_adapter_weight[key] + merged_task_vector[key]

torch.save(saved_adapter_weight, os.path.join(output_dir, 'adapter_model.bin'))

print(f"Save the: ({task_vectors_merged_methods}) merged adapter weight to {saved_adapter_weight_path}")

