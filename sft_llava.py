import os
import argparse
import tqdm
import copy
import json

from configs import hf_token, HF_CACHE, llm_domains
os.environ['HF_HOME'] = HF_CACHE
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from collections import defaultdict

from configs import RESULT_DIR
from transformers import get_scheduler
from data_generator.inference_loader import load_infer_open_data, load_infer_mc_data
from model_helper import calc_metric
from peft import LoraConfig, get_peft_model
from configs import prompt_formats, system_message
from data_generator.data_loader import DataCreator
from trl import SFTTrainer
import transformers
from trl import SFTConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText


IGNORE_INDEX = -100
model_mapper_dict = {
    0: "llava-v1.6-vicuna-7b-hf",
    1: "llava-v1.6-vicuna-13b-hf",
    2: "Qwen2.5-VL-7B-Instruct",
    3: "InternVL2-8B",
    4: "deepseek-vl2-tiny",
    5: "deepseek-vl2-small"
}

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="results")
    cache_dir: Optional[str] = field(default=None)
    learning_rate: float = field(default=0.00005)
    save_steps: float = field(default=500)
    logging_steps: float = field(default=100)
    num_train_epochs: float = field(default=3)
    # weight_decay: float = field(default=0.01),
    # optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=300,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )



class InstructDataset(Dataset):
    def __init__(self, model_outputs, questions, in_label, images) -> None:
        super().__init__()
        self.model_outputs = model_outputs
        self.questions = questions
        self.in_label = in_label
        self.images = images

    def __len__(self):
        return len(self.in_label)

    def __getitem__(self, index):
        return dict(model_outputs=self.model_outputs[:, index], 
                    questions=self.questions[index],
                    labels=self.in_label[index],
                    images=self.images[index])

class DataCollatorForInstructDataset:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        merged_dict = defaultdict(list)
        for d in examples:
            for key, value in d.items():
                merged_dict[key].append(value)
        merged_dict["model_outputs"] = np.array(merged_dict["model_outputs"]).T
        data_dict = tokenize_inputs(
            self.processor, **merged_dict
        )
        # data_dict["attention_mask"] = data_dict["input_ids"].ne(self.processor.tokenizer.pad_token_id).int()
        return data_dict



def tokenize_inputs(processor, model_outputs, questions, labels, images, return_labels=True):
    M, N = model_outputs.shape
    ens_prompts = []
    for i in range(N):
        empty_prompt = prompt_formats["ensemble_instruct_format"]
        text = ""
        for j in range(M):
            text += f"model-{j+1}: {model_outputs[j, i]}\n"
        # empty_prompt = empty_prompt.format(questions[i], text)
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": images[i],
                    },
                    {
                        "type": "text",
                        "text": questions[i] + f"Candidate Model outputs:\n{text}"
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": labels[i]}],
            },
        ]
        if not return_labels:
            conversation = conversation[:2]
        prompt = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        ens_prompts.append(prompt)

    # Tokenize the texts and process the images
    batch = processor(
        text=ens_prompts, images=images, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    if return_labels:
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

    return batch


def check_im_size(image):
    new_width = max(image.width, 28)
    new_height = max(image.height, 28)

    # Resize the image
    image = image.resize((new_width, new_height), Image.BILINEAR)
    return image


def get_images(task_name, ds_split, num_samples):
    ds_creator = DataCreator(task_name)
    images = []
    for ds in ds_creator.get(ds_split):
        for example in tqdm.tqdm(ds):
            for _ in example["question"]:
                im = example["image"].convert("RGB")
                images.append(im)
                if len(images) == num_samples:
                    return images


def run(args, training_args):
    np.random.seed(args.seed)
    model_names = [model_mapper_dict[int(i)] for i in args.model_ids]
    input_dir = os.path.join(RESULT_DIR, args.task_name)
    # model_path = f"llava-hf/llava-v1.6-vicuna-7b-hf"
    # model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    # model_path = "qwen2-7b-instruct-trl-sft-ChartQA/checkpoint-279"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_outs, train_q, train_lbl = load_infer_open_data(model_names, args.task_name, ds_split="train")
    val_outs, val_q, val_lbl = load_infer_open_data(model_names, args.task_name, ds_split="validation")
    test_outs, test_q, test_lbl = load_infer_open_data(model_names, args.task_name, ds_split="test")

    # processor = LlavaNextProcessor.from_pretrained(model_path)
    # model = LlavaNextForConditionalGeneration.from_pretrained(
    #     model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, token=hf_token
    # )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    # min_pixels = 256 * 28 * 28
    # max_pixels = 1280 * 28 * 28
    # processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    # model = AutoModelForImageTextToText.from_pretrained(model_path)
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", 
    )

    train_images = get_images(args.task_name, "train", len(train_lbl))
    val_images = get_images(args.task_name, "validation", len(val_lbl))
    test_images = get_images(args.task_name, "test", len(test_lbl))

    train_dataset = InstructDataset(train_outs, train_q, train_lbl, train_images)
    val_dataset = InstructDataset(val_outs, val_q, val_lbl, val_images)
    test_dataset =  InstructDataset(test_outs, test_q, test_lbl, test_images)

    data_collator = DataCollatorForInstructDataset(processor)

    model.train()
    # model.print_trainable_parameters()
    sft_args = SFTConfig(
        output_dir="qwen2-7b-instruct-trl-sft-ChartQA",  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=30,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=60,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        # push_to_hub=True,  # Whether to push model to Hugging Face Hub
        # report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
        dataloader_pin_memory=False
    )
    sft_args.remove_unused_columns=False
    trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=val_dataset,
            # dataset_text_field="", # needs dummy value
            peft_config=peft_config,
            tokenizer=processor.tokenizer,
        )
    trainer.train()

    trainer.save_state()
    final_model_dir = os.path.join("results", "ens_models", f"llava_instruct")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)

    # logging.info(f"Saving model into {final_model_dir}")
    model.save_pretrained(final_model_dir)

    print("Generating test outputs...")
    test_outs = []
    model.to("cuda")
    for test_sample in tqdm.tqdm(test_dataset):
        test_sample["model_outputs"] = np.expand_dims(test_sample["model_outputs"], 1)
        test_sample["questions"] = [test_sample["questions"]]
        test_sample["labels"] = [test_sample["labels"]]
        test_sample["images"] = [test_sample["images"]]
        data_dict = tokenize_inputs(processor, **test_sample, return_labels=False)
        data_dict = {k: v.to("cuda") for k, v in data_dict.items()}
        # data_dict["attention_mask"] = data_dict["input_ids"].ne(processor.tokenizer.pad_token_id).int()
        output = model.generate(
            **data_dict, 
            max_new_tokens=30, 
            temperature=0.1, 
            return_dict_in_generate=True, 
            output_scores=True
            )
        output_txt = processor.decode(output["sequences"][0], skip_special_tokens=True)
        # print(output_txt)
        # output_txt = output_txt.split("ASSISTANT: ")[-1]
        test_outs.append(output_txt.split("assistant\n")[-1])
    
    output_path = "test_outputs_more_trained_splitted.json"

    # Save the outputs to a JSON file
    with open(output_path, 'w') as f:
        json.dump(test_outs, f, indent=4)

    print(f"Test outputs saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--task_name", type=str, default="ocr",
                        choices=["mmmu"])
    parser.add_argument('--model_ids', default="123", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    arguments = parser.parse_args()
    parser = transformers.HfArgumentParser((TrainingArguments))
    train_args = parser.parse_args_into_dataclasses()
    run(arguments, train_args)

