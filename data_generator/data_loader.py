import os
import sys
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from configs import DATA_DIR, hf_token, HF_CACHE
os.environ['HF_HOME'] = HF_CACHE

from datasets import load_dataset

class DataCreator:
    data_domains = {
        "ocr": "howard-hou/OCR-VQA",
        "okvqa": "HuggingFaceM4/A-OKVQA",
        "mmmu": "MMMU/MMMU",
        "mmmu_pro": "MMMU/MMMU_Pro"
    }
    
    mmmu_subsets = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering',
        'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 
        'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design',
        'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
        'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 
        'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
        'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    
    
    def __init__(self, dataset_name):
        assert(dataset_name in self.data_domains.keys())
        self.dataset_name = dataset_name
    
        self.ds = []
        if "mmmu" not in dataset_name:
            print(f"Loading {dataset_name}")
            ds = load_dataset(self.data_domains[dataset_name], token=hf_token)
            if dataset_name == "ocr":
                ds = ds.rename_column("answers", "answer")
                ds = ds.rename_column("questions", "question")
            else:
                ds = ds.rename_column("correct_choice_idx", "answer")
            self.ds.append(ds)
        elif dataset_name == "mmmu":
            pbar = tqdm.tqdm(self.mmmu_subsets, desc="Loading MMMU dataset")
            for subset_name in pbar:
                ds = load_dataset(self.data_domains[dataset_name], subset_name, token=hf_token)
                self.ds.append(ds)
                pbar.set_postfix(status=f"Loading {subset_name}...")
        else:
            self.ds.append(
                load_dataset(self.data_domains[dataset_name],
                             "standard (4 options)", token=hf_token))

    def get(self, split_type="train"):
        for ds in self.ds:
            yield ds[split_type]


if __name__ == "__main__":
    ds_creator = DataCreator("ocr")
    for dataset_obj in ds_creator.get("validation"):
        print(len(dataset_obj))
            



