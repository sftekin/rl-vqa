import re

def replace_image_tags(text):
    return re.sub(r"<image \d>", "<image>", text)


def apply_processor(processor, text):
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt



def construct_prompt(sample, config, processor, ds_name):
    """
    sample = {
        "question":,
        "options":,
        "question_type":,
        "answer:,
    }
    
    config = {
        "multi_choice_example_format": ""
        "short_ans_example_format": ""
    }

    """
    if "mmmu" in ds_name:
        question = replace_image_tags(sample['question'])
        options = eval(sample['options'])
    else:
        question, options = sample["question"], options["options"]

    if ds_name == "mmmu":
        category = sample['question_type']
    else:
        category = 'multiple-choice'
        sample["question_type"] = category

    example = ""
    prediction_range = []
    if  category == 'multiple-choice':
        start_chr = 'A'
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
    else:
        empty_prompt_sample_structure = config['short_ans_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question)
    
    prompt = apply_processor(processor=processor, text=empty_prompt)
    res_dict = {
        "prompt": prompt,
        'correct_choice': sample["answer"],
        "prediction_range": prediction_range
    }
    return res_dict

