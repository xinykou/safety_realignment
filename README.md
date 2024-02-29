# Safety Re-Alignment of Fine-tuned Language Models through Model Fusion
![Alt text](overview.png)

## 1. SFT on downstream tasks 
### Train dataset 
- Chinese and English datasets are available on [LLaMA_Factory](https://github.com/hiyouga/LLaMA-Factory) (./llama_factory/data/alpaca_gpt4_data_en.json, ./llama_factory/data/alpaca_gpt4_data_zh.json).
- Hindi ()
- Code ()
- Math ()

### Evaluation dataset
- Chinese (XCOPA, ./lm_eval/tasks/xcopa/default_zh.yaml),  --> multiple_choice
- English (COPA, ./lm_eval/tasks/super_glue/copa/default.yaml)

````

````
## 2. Prepare safe data for training a safe mask
- Following the dataset provided in method [PKU_Beaver](https://github.com/PKU-Alignment/safe-rlhf), we prepare safety preference dataset from [PKU-Alignment/PKU-SafeRLHF-10K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K) (extra larger dataset is [PKU-Alignment/PKU-SafeRLHF-30K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-30K)). 
````

````

## 3. Train a safe mask for indicating a safe region in the weight space

````

````

## 4. Evaluation safety for realignment of fine-tuned models
### (1) single task fine-tuned models
````

````
### (2) fusion model fine-tuned on multi-task datasets
````

````