import os
import time
import tqdm
import glob
import argparse
from configs import hf_token, HF_CACHE, llm_domains
os.environ['HF_HOME'] = HF_CACHE

import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs import hf_token, prompt_formats
import torch.nn.functional as F

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from data_generator.data_loader import DataCreator
from data_generator.data_helper import construct_prompt


# class EosListStoppingCriteria(StoppingCriteria):
#     """
#     ### Instruction: -> [835, 2799, 4080, 29901]
#     ###END -> [835, 11794]
#     """
#     def __init__(self, eos_sequence = [835, 2799, 4080, 29901]):
#         self.eos_sequence = eos_sequence
#         self.count = 0

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
#         # return self.eos_sequence in last_ids
#         match_ratio = 0
#         for t in self.eos_sequence:
#             if t in last_ids[0]:
#                 match_ratio += 1
#         match_ratio /= len(self.eos_sequence)
#         return match_ratio > 0


# def get_checkpoint_number(save_dir):
#     out_files = [f for f in glob.glob(os.path.join(save_dir, "*.csv")) if "final" not in f]
#     if len(out_files) > 0:
#         file_counts = [int(os.path.basename(f).replace(".csv", "").split("_")[1]) for f in out_files]
#         latest_number = max(file_counts)
#     else:
#         latest_number = 0
#     return latest_number 


# def get_prompts(task_name, dataset_type="test"):
#     if task_name == "safety":
#         sources, targets, questions = load_safety_dataset(dataset_type, safe_flag=False)
#     elif task_name == "truthfulness":
#         sources, targets, questions = load_truth_dataset(dataset_type)
#     elif task_name == "helpfulness":
#         sources, targets, questions = load_hfl_dataset(dataset_type)
#     else:
#         raise KeyError(task_name)
#     return sources, targets, questions


# def save_outputs(save_dir, prompts, outputs, count=None):
#     if count is not None:
#         file_name = os.path.join(save_dir, f"outputs_{count}.csv")
#     else:
#         file_name = os.path.join(save_dir, f"outputs_final.csv")
#     output_df = pd.DataFrame({"prompts": prompts[:len(outputs)],
#                                 "outputs": outputs})
#     output_df.to_csv(file_name)


def run(args):
    print(args)

    model_path = f"llava-hf/llava-v1.6-vicuna-7b-hf"

    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 

    save_dir = os.path.join("results", "inference", args.task_name, args.dataset_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = model.to("cuda")

    ds_creator = DataCreator(args.task_name)
    questions, generated_outputs, ground_truths, choice_probs = [], [], [], []
    for ds in ds_creator.get(args.dataset_type):
        print(len(ds))
        for example in tqdm.tqdm(ds):
            images = [example[f"image_{i}"] for i in range(1, 8) if example[f"image_{i}"] is not None]
            res_dict = construct_prompt(example, config=prompt_formats, processor=processor)
            if len(images) > 1 or example["question_type"] == "open":
                continue
            inputs = processor(images=images[0], text=res_dict["prompt"], return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=100, return_dict_in_generate=True, output_scores=True)
            output_txt = processor.decode(output["sequences"][0], skip_special_tokens=True)
            probs_first_token = torch.nn.functional.softmax(output["scores"][0], dim=-1)
            token_ids = [processor.tokenizer.encode(letter)[1] for letter in res_dict["prediction_range"]]
            choice_probs.append(probs_first_token[0, token_ids])
            generated_outputs.append(output_txt)
            questions.append(example["question"])
            ground_truths.append(example["answer"])

    data_df = pd.DataFrame({
        "question": questions,
        "answer": ground_truths,
        "generated_outputs": generated_outputs
    })

    # not every question has 4 choices, thus, pad the small length with 0.
    choice_probs = torch.nn.utils.rnn.pad_sequence(choice_probs, batch_first=True, padding_value=0)
    probs = choice_probs.cpu().numpy()
    output_path = os.path.join(save_dir, "output.csv")
    prob_path = os.path.join(save_dir, "prob.npy")

    data_df.to_csv(output_path)
    np.save(prob_path, probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--task_name", type=str, default="mmmu", 
                        choices=["ocr", "okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument("--model_name", type=str, default="llava-1.6-7b-hf")
    parser.add_argument("--dataset_type", type= str, default="validation", choices=["test", "validation", "train"])
    parser.add_argument("--num_samples", type=int, default=1000)
    arguments = parser.parse_args()
    run(arguments)
