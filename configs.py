DATA_DIR = "data"
PROMPTS_DIR = "prompts"
RESULT_DIR = "results"
# RESULT_DIR = "results"
# HF_CACHE = "~/.cache/huggingface"
HF_CACHE = "~/scratch/hf-cache"

hf_token = "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"

llm_domains = {
    "llava-v1.6-vicuna-7b-hf": "llava-hf",
    "llava-v1.6-vicuna-13b-hf": "llava-hf",
    "Qwen2.5-VL-7B-Instruct": "Qwen",
}


prompt_formats = {
    "multi_choice_example_format": "{}\n{}\nAnswer with the option's letter from the given choices directly.",
    "short_ans_example_format": "{}\nAnswer the question using a single word or phrase."
}

