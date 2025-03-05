import os
import glob
import numpy as np
import pandas as pd


infer_dir = "results/inference"


def load_infer_prob_data(model_names, task_name, ds_split):
    prob_data = []
    labels = None
    for mn in model_names:
        data_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_output.csv")
        data_df = pd.read_csv(data_path)

        arr_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_prob.npy")
        prob_arr = np.load(arr_path)

        start_chr = 'A'
        choices = []
        for i in range(prob_arr.shape[1]):
            choices.append(start_chr)
            start_chr = chr(ord(start_chr) + 1)

        labels = []
        answers = data_df["answer"].values.astype(str)
        for ans in answers:
            labels.append(choices.index(ans))
        labels = np.array(labels)

        if task_name == "mmmu_pro" and "llava" not in mn:
            labels = np.delete(labels, (1017), axis=0)
            prob_arr = np.delete(prob_arr, (1017), axis=0)
        
        prob_data.append(prob_arr)
    
    prob_data = np.concatenate(prob_data, axis=1)
    data = np.concatenate([prob_data, labels[:, None]], axis=1)

    return data



def load_infer_open_data(model_names, task_name, ds_split):
    model_outputs = []
    answers = []
    questions = []
    for mn in model_names:
        data_path = os.path.join(infer_dir, task_name, ds_split, f"{mn}_output.csv")
        data_df = pd.read_csv(data_path, index_col=0)
        model_outputs.append(data_df["generated_outputs"].values)
        if len(answers) == 0:
            answers = data_df["answer"].tolist()
            questions = data_df["question"].tolist()
    model_outputs = np.array(model_outputs)

    return model_outputs, questions, answers

