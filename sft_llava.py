import os
import argparse
import tqdm
import copy

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
from configs import prompt_formats
from data_generator.data_loader import DataCreator
from trl import SFTTrainer
import transformers
from trl import SFTConfig


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
        data_dict["attention_mask"] = data_dict["input_ids"].ne(self.processor.tokenizer.pad_token_id).int()
        return data_dict



def tokenize_inputs(processor, model_outputs, questions, labels, images):
    M, N = model_outputs.shape
    ens_prompts = []
    for i in range(N):
        empty_prompt = prompt_formats["ensemble_instruct_format"]
        text = ""
        for j in range(M):
            text += f"Model-{j+1} thinks that the answer is {model_outputs[j, i]}\n"
        empty_prompt = empty_prompt.format(questions[i], text)
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text", 
                    "text": empty_prompt
                },
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        ens_prompts.append(prompt)

    combined_text = [f"{s} {t}" for (s, t) in zip(ens_prompts, labels)]
    model_inputs = processor(
                images=images, 
                text=combined_text, 
                return_tensors="pt",
                padding="longest",
                truncation=True).to("cuda")
    input_ids = model_inputs["input_ids"]
    lbl = copy.deepcopy(input_ids)

    source_tokens = processor(
            images=images, 
            text=ens_prompts, 
            return_tensors="pt",
            padding=True,
            truncation=True).to("cuda")
    target_indices = source_tokens["input_ids"].ne(processor.tokenizer.pad_token_id).sum(dim=1)
    for i, idx in enumerate(target_indices):
        lbl[i, :idx] = IGNORE_INDEX
    lbl[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=lbl)


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
    model_path = f"llava-hf/llava-v1.6-vicuna-7b-hf"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_outs, train_q, train_lbl = load_infer_open_data(model_names, args.task_name, ds_split="train")
    val_outs, val_q, val_lbl = load_infer_open_data(model_names, args.task_name, ds_split="validation")
    test_outs, test_q, test_lbl = load_infer_open_data(model_names, args.task_name, ds_split="test")

    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, token=hf_token
    )
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", 
    )


    train_images = get_images(args.task_name, "train", len(train_lbl))
    # val_images = get_images(args.task_name, "validation", len(val_lbl))
    # test_images = get_images(args.task_name, "test", len(test_lbl))

    train_dataset = InstructDataset(train_outs, train_q, train_lbl, train_images)
    # val_dataset = InstructDataset(val_outs, val_q, val_lbl, val_images)
    # test_dataset =  InstructDataset(test_outs, test_q, test_lbl, test_images)

    data_collator = DataCollatorForInstructDataset(processor)

    model.train()
    # model.print_trainable_parameters()
    sft_args = SFTConfig(
        output_dir="qwen2-7b-instruct-amazon-description", # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=4,          # batch size per device during training
        gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=5,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-5,                     # learning rate, based on QLoRA paper
        # bf16=True,                              # use bfloat16 precision
        fp16=True,
        tf32=True,                              # use tf32 precision
        max_grad_norm=None,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
        # dataset_text_field="", # need a dummy field for collator
        dataset_kwargs = {"skip_prepare_dataset": True}, # important for collator
        dataloader_pin_memory=False
    )
    sft_args.remove_unused_columns=False
    trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            # dataset_text_field="", # needs dummy value
            peft_config=peft_config,
            tokenizer=processor.tokenizer,
        )
    trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--task_name", type=str, default="ocr",
                        choices=["mmmu"])
    parser.add_argument('--model_ids', default="123", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    arguments = parser.parse_args()
    parser = transformers.HfArgumentParser((TrainingArguments))
    train_args = parser.parse_args_into_dataclasses()
    run(arguments, train_args)

