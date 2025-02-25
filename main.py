import argparse
import transformers

from data_generator.inference_loader import load_infer_prob_data



def run():
    task_name = "mmmu_pro"
    ds_split = "test"


    model_names = ["llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf",
                "Qwen2.5-VL-7B-Instruct", "InternVL2-8B",
                "deepseek-vl2-tiny", "deepseek-vl2-small"]


    pred_data, label = load_infer_prob_data(model_names, task_name, ds_split)


if __name__ == "__main__":
    run()
