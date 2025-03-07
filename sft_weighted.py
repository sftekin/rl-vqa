import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import RESULT_DIR
from data_generator.inference_loader import load_infer_prob_data


model_mapper_dict = {
    0: "llava-v1.6-vicuna-7b-hf",
    1: "llava-v1.6-vicuna-13b-hf",
    2: "Qwen2.5-VL-7B-Instruct",
    3: "InternVL2-8B",
    4: "deepseek-vl2-tiny",
    5: "deepseek-vl2-small"
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], output_dim),
        )
        self.net.apply(self.init_weights)

    def forward(self, x):
        out = self.net(x)
        out = torch.softmax(out, dim=-1)
        return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


def train_ensemble(model_names, train_loader, val_loader, novel_loader, n_epochs, save_dir, space_size, verbose=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = MLP(len(model_names) * space_size, [100, 100], space_size)
    model = model.to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc, tol = (0, 0)
    for epoch in range(n_epochs):
        avg_loss = []
        for i, batch_data in enumerate(train_loader):
            in_x = batch_data[:, :-1].to("cuda").float()
            label = batch_data[:, -1].type(torch.long).to("cuda")

            optimizer.zero_grad()
            out = model(in_x)
            loss = loss_fn(out, label)

            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        if epoch % 10 == 0 and verbose:
            run_loss = np.mean(avg_loss)
            print(f'Epoch {epoch} | Loss {run_loss:.4f}')

        if val_loader:
            acc_mean = test_loop(model, val_loader)

            if acc_mean > best_val_acc:
                outfile = os.path.join(save_dir, f'best_model.tar')
                torch.save({'epoch': epoch,
                            'state': model.state_dict(),
                            "accuracy": acc_mean}, outfile)
                best_val_acc = acc_mean
                tol = 0
            else:
                tol += 1

            if tol > 300:
                print("No improvement in 200 epochs, breaking")
                break

    if val_loader:
        best_dict = torch.load(f"{save_dir}/best_model.tar")
        model.load_state_dict(best_dict["state"])

    model.eval()
    acc_mean, logits, labels = test_loop(model, novel_loader, ret_logit=True)
    print(f'Novel Acc = {acc_mean:.4f}')
    exp_result = dict(val_acc=best_dict["accuracy"],
                      test_acc=acc_mean,
                      state=model.state_dict(),
                      model_names=model_names,
                      logits=logits,
                      labels=labels)

    output_path = os.path.join(save_dir, "exp_result.pth")
    torch.save(exp_result, output_path)

    return exp_result


def test_loop(model, data_loader, ret_logit=False, device="cuda"):
    assert device in ["cuda", "cpu"]
    acc_all = []
    logits = []
    labels = []
    for i, batch_data in enumerate(data_loader):
        in_x = batch_data[:, :-1].to(device).float()
        scores = model(in_x)
        label = batch_data[:, -1].numpy()

        scores = scores.detach().cpu().numpy()
        in_x = in_x.detach().cpu().numpy()
        pred = np.argmax(scores, axis=1)
        corrects = np.sum(pred == label)
        acc_all.append(corrects / len(label) * 100)
        if ret_logit:
            logits.append(np.concatenate([in_x, scores], axis=1))
            labels.append(label)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)

    if ret_logit:
        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        return acc_mean, logits, labels
    else:
        return acc_mean


def run(args):
    np.random.seed(args.seed)
    model_names = [model_mapper_dict[int(i)] for i in args.model_ids]

    data = load_infer_prob_data(model_names, args.task_name, args.dataset_type)
    space_size = (data.shape[1] - 1) // len(model_names)

    test_data = load_infer_prob_data(model_names, "okvqa", "validation")
    # test_data = load_infer_prob_data(model_names, "okvqa", "validation")

    rand_idx = np.random.permutation(len(data))
    data = data[rand_idx]
    ds_len = len(data)
    train_size = int(ds_len * 0.7)
    val_size = int(ds_len * 0.3)
    split = {"train": data[:train_size],
             "val": data[train_size:train_size + val_size],
             "test": test_data}

    train_loader = DataLoader(split["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(split["val"], batch_size=args.batch_size, shuffle=True)
    novel_loader = DataLoader(split["test"], batch_size=args.batch_size, shuffle=False)

    train_ensemble(model_names, train_loader, val_loader, novel_loader,
                   n_epochs=300, save_dir="results/ensemble",
                   space_size=space_size, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_name", type=str, default="okvqa",
                        choices=["okvqa", "mmmu", "mmmu_pro"])
    parser.add_argument('--model_ids', default="123", type=str)
    parser.add_argument("--dataset_type", type= str, default="train", 
                        choices=["test", "validation", "train"])
    parser.add_argument('--batch_size', default=16, type=int)
    arguments = parser.parse_args()
    run(arguments)

