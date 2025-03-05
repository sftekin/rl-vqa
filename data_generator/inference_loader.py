import re
import os
import glob
import numpy as np
import pandas as pd


from data_generator.data_loader import DataCreator
from configs import prompt_formats


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


def load_infer_mc_data(model_names, task_name, ds_split):
    ds_creator = DataCreator(task_name)
    ds_questions= []
    for ds in ds_creator.get(ds_split):
        for example in ds:
            start_chr = 'A'
            index2ans = {}
            option_txt = ""
            prediction_range = []
            if "mmmu" in task_name:
                options = eval(example["options"])
            else:
                options = example["options"]
            for option in options:
                prediction_range.append(start_chr)
                option_txt += f"({start_chr}) {option}\n"
                index2ans[start_chr] = option
                start_chr = chr(ord(start_chr) + 1)
            empty_prompt_sample_structure = prompt_formats['multi_choice_example_format']
            empty_prompt = empty_prompt_sample_structure.format(example["question"], option_txt)
            ds_questions.append(empty_prompt)

    def extract_letter(text):
        match = re.search(r"\((\w)\)", text)
        return match.group(1) if match else ""

    model_outputs, answers = [], []
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

        generated_outputs = data_df["generated_outputs"].values
        if len(answers) == 0:
            answers = data_df["answer"].tolist()

        extracted_outputs = []
        for output in generated_outputs:
            pred_txt = str(output)[:10].strip()
            if "\n" in pred_txt:
                pred_txt = pred_txt.split("\n")[1]
            if "(" in pred_txt or ")" in pred_txt:
                pred_txt = extract_letter(pred_txt)
            extracted_outputs.append(pred_txt[:1].upper())
        extracted_outputs = np.array(extracted_outputs)

        labels = data_df["answer"].values.astype(str) 
        if task_name == "mmmu_pro" and "llava" not in mn:
            extracted_outputs = np.delete(extracted_outputs, (1017), axis=0)
            labels = np.delete(labels, (1017), axis=0)
            ds_questions = np.delete(ds_questions, (1017), axis=0)
        model_outputs.append(extracted_outputs)
    model_outputs = np.array(model_outputs)
    ds_questions = ds_questions[:len(model_outputs[0])]
    return model_outputs, ds_questions, answers

