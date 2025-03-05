import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

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
from PIL import Image

from configs import hf_token, prompt_formats, llm_domains
import torch.nn.functional as F

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from data_generator.data_loader import DataCreator
from data_generator.data_helper import construct_open_ended_prompt
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoModel, AutoTokenizer
from model_helper import load_image


def save_checkpoint(save_dir, model_name, data_dict, step_num=None):
    data_df = pd.DataFrame({
        "question":data_dict["questions"],
        "answer": data_dict["ground_truths"],
        "generated_outputs": data_dict["generated_outputs"]
    })

    if step_num is None:
        output_path = os.path.join(save_dir, f"{model_name}_output.csv")
    else:
        output_path = os.path.join(save_dir, f"{model_name}_output_{step_num}.csv")

    # save model
    data_df.to_csv(output_path)


def check_im_size(image):
    new_width = max(image.width, 28)
    new_height = max(image.height, 28)

    # Resize the image
    image = image.resize((new_width, new_height), Image.BILINEAR)
    return image


def load_model(model_path):
    if "llava" in model_path:
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, token=hf_token)
    elif "Qwen" in model_path:
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        model = AutoModelForImageTextToText.from_pretrained(model_path)
    elif "InternVL2" in model_path:
        model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True).eval().cuda()
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

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
    questions, generated_outputs, ground_truths = [], [], []
    max_num_samples = args.num_samples
    num_samples = 0
    question_type = "open"
    for ds in tqdm.tqdm(ds_creator.get(args.dataset_type), total=len(ds_creator)):
        for example in tqdm.tqdm(ds):
            images = [example["image"]]
            for question, answer in zip(example["question"], example["answer"]):

                if "InternVL2" not in args.model_name:
                    res_dict = construct_open_ended_prompt(question, prompt_formats, processor=processor)
                    inputs = processor(
                        images=check_im_size(images[0]), 
                        text=[res_dict["prompt"]], 
                        return_tensors="pt").to("cuda")
                    output = model.generate(
                        **inputs, 
                        max_new_tokens=100, 
                        temperature=0.7, 
                        return_dict_in_generate=True, 
                        output_scores=True
                        )
                    output_txt = processor.decode(output["sequences"][0], skip_special_tokens=True)
                    if "Qwen2.5" in args.model_name:
                        output_txt = output_txt.split("\n")[-1]
                    elif "llava" in args.model_name:
                        output_txt = output_txt.split("ASSISTANT: ")[-1]
                else:
                    res_dict = construct_open_ended_prompt(question, prompt_formats, processor=None)
                    generation_config = dict(max_new_tokens=100, return_dict_in_generate=True, output_scores=True)
                    pixel_values = load_image(images[0]).to(torch.bfloat16).cuda()
                    output = model.chat(processor, pixel_values, res_dict["prompt"], generation_config)
                    output_txt = processor.decode(output["sequences"][0], skip_special_tokens=True)

                generated_outputs.append(output_txt)
                questions.append(question)
                ground_truths.append(answer)
                num_samples += 1

                if num_samples % 500 == 0 and num_samples > 0:
                    data_dict = {
                        "questions": questions,
                        "ground_truths": ground_truths,
                        "generated_outputs": generated_outputs,
                    }
                    save_checkpoint(save_dir, args.model_name, data_dict, num_samples)
                if num_samples > max_num_samples:
                    break
            if num_samples > max_num_samples:
                break
        if num_samples > max_num_samples:
            break

    # Final Save
    data_dict = {
        "questions": questions,
        "ground_truths": ground_truths,
        "generated_outputs": generated_outputs,
    }
    save_checkpoint(save_dir, args.model_name, data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--task_name", type=str, default="ocr", 
                        choices=["ocr", "okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument("--model_name", type=str, default="llava-v1.6-vicuna-13b-hf",
                        choices=["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "Qwen2.5-VL-7B-Instruct", "InternVL2-8B"])
    parser.add_argument("--dataset_type", type= str, default="train", choices=["test", "validation", "train"])
    parser.add_argument("--num_samples", type=int, default=2000)
    arguments = parser.parse_args()
    run(arguments)
