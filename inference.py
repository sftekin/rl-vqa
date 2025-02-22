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

from configs import hf_token, prompt_formats, llm_domains
import torch.nn.functional as F

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from data_generator.data_loader import DataCreator
from data_generator.data_helper import construct_prompt


def load_model(model_path):
    if "llava" in model_path:
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, token=hf_token) 
    return processor, model


def run(args):
    print(args)
    if args.task_name == "mmmu_pro":
        assert args.dataset_type == "test"

    model_path = f"{llm_domains[args.model_name]}/{args.model_name}"
    processor, model = load_model(model_path)

    save_dir = os.path.join("results", "inference", args.task_name, args.dataset_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = model.to("cuda")

    ds_creator = DataCreator(args.task_name)
    questions, generated_outputs, ground_truths, choice_probs = [], [], [], []
    max_num_samples = args.num_samples
    num_samples = 0
    for ds in tqdm.tqdm(ds_creator.get(args.dataset_type), total=len(ds_creator)):
        for example in tqdm.tqdm(ds):
            if "mmmu" in args.task_name:
                images = [example[f"image_{i}"] for i in range(1, 8) if example[f"image_{i}"] is not None]
            else:
                images = [example["image"]]
            res_dict = construct_prompt(
                example, config=prompt_formats, processor=processor, ds_name=args.task_name)
            if (len(images) != 1 or example["question_type"] == "open" 
                or res_dict["prompt"].count("<image>") != 1):
                continue
            inputs = processor(images=images[0], text=res_dict["prompt"], return_tensors="pt").to("cuda")
            output = model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.7, 
                return_dict_in_generate=True, 
                output_scores=True
                )
            output_txt = processor.decode(output["sequences"][0], skip_special_tokens=True)
            output_txt = output_txt[len(res_dict["prompt"]) - 7:]
            probs_first_token = torch.nn.functional.softmax(output["scores"][0], dim=-1)
            token_ids = [processor.tokenizer.encode(letter)[1] for letter in res_dict["prediction_range"]]
            choice_probs.append(probs_first_token[0, token_ids])
            generated_outputs.append(output_txt)
            questions.append(example["question"])
            if "okvqa" == args.task_name:
                ans = res_dict["prediction_range"][example["answer"]]
            else:
                ans = example["answer"]
            ground_truths.append(ans)
            num_samples += 1
            if num_samples > max_num_samples:
                break
        if num_samples > max_num_samples:
            break

    data_df = pd.DataFrame({
        "question": questions,
        "answer": ground_truths,
        "generated_outputs": generated_outputs
    })

    # not every question has 4 choices, thus, pad the small length with 0.
    choice_probs = torch.nn.utils.rnn.pad_sequence(
        choice_probs, batch_first=True, padding_value=0).cpu().numpy()
    output_path = os.path.join(save_dir, f"{args.model_name}_output.csv")
    prob_path = os.path.join(save_dir, f"{args.model_name}_prob.npy")

    # save model
    data_df.to_csv(output_path)
    np.save(prob_path, choice_probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--task_name", type=str, default="okvqa", 
                        choices=["ocr", "okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument("--model_name", type=str, default="llava-v1.6-vicuna-13b-hf",
                        choices=["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf"])
    parser.add_argument("--dataset_type", type= str, default="validation", choices=["test", "validation", "train"])
    parser.add_argument("--num_samples", type=int, default=1500)
    arguments = parser.parse_args()
    run(arguments)
