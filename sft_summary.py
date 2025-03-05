import os
import argparse
import tqdm

from configs import hf_token, HF_CACHE, llm_domains
os.environ['HF_HOME'] = HF_CACHE

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from configs import RESULT_DIR
from transformers import get_scheduler
from data_generator.inference_loader import load_infer_open_data, load_infer_mc_data
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model_helper import calc_metric


model_mapper_dict = {
    0: "llava-v1.6-vicuna-7b-hf",
    1: "llava-v1.6-vicuna-13b-hf",
    2: "Qwen2.5-VL-7B-Instruct",
    3: "InternVL2-8B",
    4: "deepseek-vl2-tiny",
    5: "deepseek-vl2-small"
}


class MyDataset(Dataset):
    def __init__(self, tokenized_inputs, labels, global_attention_tokens=None):
        self.tokenized_inputs = tokenized_inputs
        self.labels = labels
        self.global_attention_tokens = global_attention_tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokenized_inputs[idx].ids)
        attention_mask = torch.tensor(self.tokenized_inputs[idx].attention_mask)
        global_attentions = []
        start = False
        for i in input_ids:
            if start:
                if i == 50266:
                    start = False
                global_attentions.append(1)
            else:
                if i == 50265:
                    start = True
                global_attentions.append(0)
        global_attentions = torch.tensor(global_attentions)
        label = self.labels[idx]
        return_dict = {'input_ids': input_ids,
                       "labels": label,
                       'attention_mask': attention_mask,
                       'global_attention_mask': global_attentions}
        return return_dict




def tokenize_inputs(tokenizer, in_data, questions, in_label, skip_model_outs=False):
    if len(in_data.shape) == 3:
        M, N, K = in_data.shape
    elif len(in_data.shape) == 2:
        M, N = in_data.shape
        in_data = np.expand_dims(in_data, -1)
        K = 1
    else:
        M, N, K = (1, 0, 0)

    data = []
    for i in range(N):
        # create an input
        temp = [f"[BOQ]{questions[i]}[EOQ]"]
        if not skip_model_outs:
            for j in range(M):
                for k in range(K):
                    candidate_txt = str(in_data[j, i, k]).strip()
                    temp.append(f"[BOC{j}]{candidate_txt}[EOC{j}]")
        data.append("".join(temp))

    # add new tokens
    new_tokens = []
    num_added = 0
    vocab = tokenizer.get_vocab()
    for i in range(M):
        if f"[BOQ]" not in vocab and f"[BOQ]" not in new_tokens:
            new_tokens.append("[BOQ]")
            num_added += 1
        if f"[EOQ]" not in vocab and f"[EOQ]" not in new_tokens:
            new_tokens.append("[EOQ]")
            num_added += 1
        if f"[BOC{i}]" not in vocab and f"[BOC{i}]" not in new_tokens:
            new_tokens.append(f"[BOC{i}]")
            num_added += 1
        if f"[EOC{i}]" not in vocab and f"[EOC{i}]" not in new_tokens:
            new_tokens.append(f"[EOC{i}]")
            num_added += 1
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print("We have added", num_added_toks, "tokens")
    new_token_ids = [tokenizer.encode(tkn)[1] for tkn in new_tokens]

    model_inputs = tokenizer(data, padding="longest", max_length=16000, truncation=True, return_tensors="pt")
    lbl_ids = tokenizer(in_label, padding="longest", max_length=512, truncation=True, return_tensors="pt")

    return model_inputs, lbl_ids, new_token_ids


def extract_answer(tokenizer, prediction):
    batch_size = prediction.shape[0]
    pred = []
    for i in range(batch_size):
        answer_txt = tokenizer.decode(prediction[i], skip_special_tokens=True)
        pred.append(answer_txt.strip())
    return pred



def test_loop(model, tokenizer, eval_dataloader, device, mode="Validation", return_outputs=False):
    model.eval()
    predictions, labels = [], []
    progress_bar = tqdm.tqdm(range(len(eval_dataloader)))
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        predictions.append(extract_answer(tokenizer, pred))
        labels.append(extract_answer(tokenizer, batch["labels"]))
        progress_bar.update(1)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    scores = calc_metric(labels, predictions)
    print(f"{mode} BLUE-1: {scores[0]:.4f} EM: {scores[1]:.4f} F1: {scores[2]:.4f}")


    if return_outputs:
        return scores, predictions, labels
    else:
        return scores


def run(args):
    np.random.seed(args.seed)
    model_names = [model_mapper_dict[int(i)] for i in args.model_ids]
    input_dir = os.path.join(RESULT_DIR, args.task_name)
    ens_model_n = "allenai/led-base-16384"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_outs, train_q, train_lbl = load_infer_mc_data(model_names, "okvqa", ds_split="train")

    num_train_samples = len(train_lbl)
    train_size = int(num_train_samples * 0.7)
    val_outs, val_q, val_lbl = train_outs[:, train_size:], train_q[train_size:], train_lbl[train_size:]
    train_outs, train_q, train_lbl = train_outs[:, :train_size], train_q[:train_size], train_lbl[:train_size]

    test_outs, test_q, test_lbl = load_infer_mc_data(model_names, "okvqa", ds_split="validation")

    # train_outs, train_q, train_lbl = load_infer_open_data(model_names, args.task_name, ds_split="train")
    # val_outs, val_q, val_lbl = load_infer_open_data(model_names, args.task_name, ds_split="validation")
    # test_outs, test_q, test_lbl = load_infer_open_data(model_names, args.task_name, ds_split="test")

    tokenizer = AutoTokenizer.from_pretrained(ens_model_n)
    train_inputs, train_labels, new_token_ids = tokenize_inputs(tokenizer, train_outs, train_q, train_lbl,
                                                                skip_model_outs=False)
    val_inputs, val_labels, _ = tokenize_inputs(tokenizer, val_outs, val_q, val_lbl,
                                                            skip_model_outs=False)
    test_inputs, test_labels, _ = tokenize_inputs(tokenizer, test_outs, test_q, test_lbl,
                                                  skip_model_outs=False)


    train_dataset = MyDataset(train_inputs, train_labels.input_ids,
                              global_attention_tokens=new_token_ids)
    val_dataset = MyDataset(val_inputs, val_labels.input_ids,
                            global_attention_tokens=new_token_ids)
    test_dataset = MyDataset(test_inputs, test_labels.input_ids,
                             global_attention_tokens=new_token_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(ens_model_n)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.to(device)

    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm.tqdm(range(num_training_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    best_dict = model.state_dict()
    best_val_score, tol, determining_score_idx = 0, 0, 0
    for epoch in range(args.num_epochs):
        running_loss1 = []
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            running_loss1.append(loss.item())
            progress_bar.set_postfix({"Train Loss": np.mean(running_loss1)})
            torch.cuda.empty_cache()

        scores = test_loop(model, tokenizer, val_loader, device)
        acc_mean = scores[determining_score_idx]
        if acc_mean > best_val_score:
            best_val_score = acc_mean
            best_dict = model.state_dict()
            tol = 0
        else:
            tol += 1

        if tol >= 3:
            print("early stopping...")
            break

    print("Testing model performance")
    model.load_state_dict(best_dict)
    test_scores = test_loop(model, tokenizer, test_loader, device, mode="Test")
    score_str = f"Combinations {args.model_ids} \t scores:{test_scores}\n"
    comb_code = "".join(map(lambda x: str(x), args.model_ids))
    scores_path = os.path.join("results", f"scores_{args.task_name}_{comb_code}.txt")
    with open(scores_path, "a") as f:
        f.write(score_str)

    print("Saving model...")
    model_save_path = os.path.join("results", "ens_models",
                                   f"best_result_{args.task_name}_{comb_code}")
    model.save_pretrained(model_save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--task_name", type=str, default="ocr",
                        choices=["mmmu"])
    parser.add_argument('--model_ids', default="123", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    arguments = parser.parse_args()
    run(arguments)

