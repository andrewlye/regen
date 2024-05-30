from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
from tqdm import tqdm
import os

map = {0 : 'A.',
       1 : 'B.',
       2: 'C.',
       3: 'D.',
       4: 'E.'}


def format_prompt(sample, instruction):
    map = {0 : 'A.',
       1 : 'B.',
       2: 'C.',
       3: 'D.',
       4: 'E.'}
    
    prompt = f"Question: {sample['question']}\n"
    for i in range(len(sample['choices'])):
        prompt += f"\t{map[i]} {sample['choices'][i]}\n"
    prompt += f"Instruction: {instruction}\nAnswer:"
    return prompt

def query(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generation_output = model.generate(inputs, max_new_tokens=512)
    output=tokenizer.decode(generation_output[0])

    return output

def format_answer(resp):
    ans = resp.split("Answer:")[1].strip().lower()
    ans = ans.split("\n")[0]
    return ans

def grade_answer(sample, ans):
    return map[sample['answer']].lower() in ans

def run_eval(instruction, log_data=False):
    dataset = load_dataset("cais/mmlu", 'all', split='validation')
    model = AutoAWQForCausalLM.from_quantized('/mnt/vstor/CSE_CSDS_VXC204/aly37/quantize-hal-project/quant-models/awq/Llama-2-13b-chat-hf', fuse_layers=True)
    tokenizer = AutoTokenizer.from_pretrained('/mnt/vstor/CSE_CSDS_VXC204/aly37/quantize-hal-project/quant-models/awq/Llama-2-13b-chat-hf', trust_remote_code=True)

    log = []
    num_correct = 0

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        prompt = format_prompt(sample, instruction)
        output = query(prompt, model, tokenizer)
        ans = format_answer(output)
        if grade_answer(sample, ans):
            num_correct += 1
        if log_data:
            log.append((prompt, ans, sample['answer'], grade_answer(sample, ans)))
    
    if log_data:
        with open("/mnt/vstor/CSE_CSDS_VXC204/aly37/regen/log.txt", "w") as f:
            f.write(f"NUM_CORRECT: {num_correct}/{len(dataset)}, ACCURACY: {num_correct/len(dataset)}\n")
            for i in range(len(log)):
                f.write(f"SAMPLE {i+1}\n")
                f.write(f"{log[i][0]}\n")
                f.write(f"Output: {log[i][1]}\n")
                f.write(f"Correct answer: {map[log[i][2]]}\n")
                f.write(f"{'(correct)' if log[i][3] else '(incorrect)'}\n\n")

    return num_correct, len(dataset)

if __name__ == '__main__':
    run_eval("Let's figure it out!", True)
        



