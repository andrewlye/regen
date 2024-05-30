from benchmarks.mmlu.take_test import run_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

dataset = load_dataset("cais/mmlu", 'all', split='validation')
sample_questions = [(dataset[i]['question'], dataset[i]['choices'], dataset[i]['answer']) for i in range(0, len(dataset), 100)]
samples = ""
map = {0 : 'A.',
       1 : 'B.',
       2: 'C.',
       3: 'D.',
       4: 'E.'}

for i in range(len(sample_questions)):
    samples += f'Question: {sample_questions[i][0]}\n'
    for j in range(len(sample_questions[i][1])):
        samples += f'\t{map[j]}. {sample_questions[i][1][j]}\n'
    samples += f'Instruction: <INS>\nAnswer: {map[sample_questions[i][2]]}'
    if i != len(sample_questions) - 1:
        samples += '\n\n'

def make_prompt(log):
    log.sort()
    past_results = ""
    for i in range(len(log)):
        past_results += f"text:\n{log[i][1]}\nscore:{int(round(log[i][0]))}\n\n"
        
    question = f'''Your task is to generate an instruction <INS> that enhances the precison in solving various subject problems. Below are some previous instructions with their corresponding scores. The score ranges from 0 to 100, where a higher score indicates a better instruction.
    \n{past_results}Below are some example problems.\n\n{samples}\n\nGenerate an instruction that is different from all the instructions <INS> above, and has a higher score than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>. The instruction should be concise, effective, and generally applicable to all problems above.'''
    
    prompt = f'<|user|>\n{question}<|end|>\n<|assistant|>'

    return prompt

def gen(prompt: str, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
    outputs = model.generate(inputs, 
                            do_sample=True,
                            temperature=1,
                            max_new_tokens=500)
    text = tokenizer.decode(outputs[0])
    return text

parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=int, required=True, default=15)
parser.add_argument('--o', type=str, required=True, default='output.txt')
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-128k-instruct', trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-128k-instruct', trust_remote_code=True)

NUM_ITERS = args.iters
log = [(45, "Let's think step by step"), (48, "Let's figure it out!")]

prompt_log = []
gen_instructs = []
accuracies = []

time_start = datetime.now()
for i in range(NUM_ITERS):
    print(f"Iteration {i+1}")
    prompt = make_prompt(log)
    prompt_log.append(prompt)
    while True:
        output = gen(prompt, model, tokenizer)
        try:
            response = output.split("<|assistant|>")[1]
            instruction = response.split("<INS>")[1].split("</INS>")[0]
            instruction = instruction.replace("<|end|>", "")
            instruction = instruction.strip()
            break
        except:
            print("Generation failed. Retrying...")
            continue
    gen_instructs.append(instruction)

    num_correct, total = run_eval(instruction)
    acc = float(num_correct) / total
    accuracies.append(acc)

    log.append((acc*100, instruction))

    time_end = datetime.now()
    delta_time = (time_end - time_start).total_seconds()
    delta_time = delta_time / 3600

    print("Writing results...")
    with open(f"/mnt/vstor/CSE_CSDS_VXC204/aly37/regen/{args.o}", "w") as f:
        f.write(f"time elapsed: {delta_time:2f} hours\n")
        f.write(f"best acc.: {max(accuracies)} from iter {np.argmax(accuracies)}\n\n")
        for iter, tup in enumerate(zip(prompt_log, gen_instructs, accuracies)):
            f.write(f"Iteration {iter+1}\n")
            f.write(f"Task prompt:\n{tup[0]}\n")
            f.write(f"Generated instruction: {tup[1]}\n")
            f.write(f"Accuracy: {tup[2]}\n\n")








    












