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
from PIL import Image

from configs import hf_token, prompt_formats, llm_domains
import torch.nn.functional as F

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from data_generator.data_loader import DataCreator
from data_generator.data_helper import construct_prompt
from transformers import AutoProcessor, AutoModelForImageTextToText


def save_checkpoint(save_dir, model_name, data_dict, step_num=None):
    data_df = pd.DataFrame({
        "question":data_dict["questions"],
        "answer": data_dict["ground_truths"],
        "generated_outputs": data_dict["generated_outputs"]
    })

    # not every question has 4 choices, thus, pad the small length with 0.
    choice_probs = torch.nn.utils.rnn.pad_sequence(
        data_dict["choice_probs"], batch_first=True, padding_value=0).cpu().numpy()
    
    if step_num is None:
        output_path = os.path.join(save_dir, f"{model_name}_output.csv")
        prob_path = os.path.join(save_dir, f"{model_name}_prob.npy")
    else:
        output_path = os.path.join(save_dir, f"{model_name}_output_{step_num}.csv")
        prob_path = os.path.join(save_dir, f"{model_name}_prob_{step_num}.npy")

    # save model
    data_df.to_csv(output_path)
    np.save(prob_path, choice_probs)


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
            if (len(images) != 1 or example["question_type"] == "open"):
                continue
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
            output_txt = output_txt[-7:]
            probs_first_token = torch.nn.functional.softmax(output["scores"][0], dim=-1)
            token_ids = [processor.tokenizer.encode(letter)[0] for letter in res_dict["prediction_range"]]
            choice_probs.append(probs_first_token[0, token_ids])
            generated_outputs.append(output_txt)
            questions.append(example["question"])
            if "okvqa" == args.task_name:
                ans = res_dict["prediction_range"][example["answer"]]
            else:
                ans = example["answer"]
            ground_truths.append(ans)
            num_samples += 1

            data_dict = {
                "questions": questions,
                "ground_truths": ground_truths,
                "generated_outputs": generated_outputs,
                "choice_probs": choice_probs
            }

            if num_samples % 100 == 0 and num_samples > 0:
                save_checkpoint(save_dir, args.model_name, data_dict, num_samples)

            if num_samples > max_num_samples:
                break
        if num_samples > max_num_samples:
            break

    # Final Save
    save_checkpoint(save_dir, args.model_name, data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--task_name", type=str, default="mmmu_pro", 
                        choices=["ocr", "okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct",
                        choices=["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "Qwen2.5-VL-7B-Instruct"])
    parser.add_argument("--dataset_type", type= str, default="test", choices=["test", "validation", "train"])
    parser.add_argument("--num_samples", type=int, default=15000)
    arguments = parser.parse_args()
    run(arguments)
