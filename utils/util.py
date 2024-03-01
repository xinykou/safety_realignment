import concurrent.futures
import json
import os

import transformers
from torch import nn
from tqdm import tqdm


def load_json_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json_file(file_path: str, data: dict or list):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def find_json_files(dir):
    files = []
    for file in os.listdir(dir):
        suffix = file.split(".")[-1]
        if suffix == "json":
            files.append(os.path.join(dir, file))
    return files


def append_data_to_jsonl(data: dict, file_path: str):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def convert_jsonl_to_json(file_path: str, *, sort_func=None, remove=False):
    assert file_path.endswith(".jsonl"), "file path must end with .jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(i) for i in f.readlines()]
    if sort_func is not None:
        data = sorted(data, key=sort_func)
    if remove:
        os.remove(file_path)
    write_json_file(file_path[:-1], data)


def multi_thread_map(fn, data, desc="Processing", max_workers=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fn, example) for example in data]
        result = list(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc))
    return [i.result() for i in result]