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
    'InternVL2-8B': "OpenGVLab",
    'deepseek-vl2-tiny': "deepseek-ai",
    'deepseek-vl2-small': "deepseek-ai"
}


prompt_formats = {
    "multi_choice_example_format": "{}\n{}\nAnswer with the option's letter from the given choices directly.",
    "short_ans_example_format": "{}\nAnswer the question using a single word or phrase.",
    "ensemble_instruct_format": "{}\n{}\nAnswer the question using a single word or phrase."
}

system_message = """You are a Vision Language Ensemble Model specialized in interpreting visual data from chart images and candidate model outputs.
Your task is to analyze the provided chart image, candidate model outputs, and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
