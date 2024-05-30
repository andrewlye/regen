from benchmarks.haltt4llm.take_test import train
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def generate_prompt(prompt):
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map='auto', cache_dir='/home/aly37/.cache')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", cache_dir='/home/aly37/.cache')
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).input_ids.to("cuda")
    outputs = model.generate(inputs, 
                         do_sample=True,
                         num_beams = 2,
                         temperature=1.2,
                         max_length=1000,)
    
    text = tokenizer.batch_decode(outputs)[0]
    lines = text[text.index("NEW PROMPT:"):].split("\n")

    if len(lines[0]) > 12:
        return lines[0].split("NEW PROMPT: ")[1]
    elif "--" in lines[1]:
        return lines[2]
    else:
        stops = ["answer:", "output:", "response:" "new prompt:"]
        query = lines[1].split()
        resultwords = [word for word in query if word.lower() not in stops]
        result = ' '.join(resultwords)
        return result


quantization_methods = ['awq']
trivia_path = "/mnt/vstor/CSE_CSDS_VXC204/aly37/quantize-hal-project/haltt4llm/hq_trivia_questions.json"
model_dir = "/mnt/vstor/CSE_CSDS_VXC204/aly37/quantize-hal-project/quant-models"
model = "Llama-2-13b-chat-hf"
cache_dir = "/mnt/vstor/CSE_CSDS_VXC204/aly37/cache"
remote_path = "meta-llama/Llama-2-13b-chat-hf"

SYS_PROMPT = "Instruct: You are a helpful assistant that aids in the improvement of prompts for models whose weights have been quantized to a lower precision. Given a list of prior prompts and their increases in accuracy, generate a single new prompt that results in greater performance of a model when recalling factual information. Your new prompt should greatly increase the overall performance of the model."
EXAMPLES = """Prompt 1: Kindly scrutinize the weight matrix within the model, as it may contain inaccuracies. It is imperative to validate its precision and make any required modifications to guarantee optimal performance in the subsequent scenario. An interaction between a user and a helpful assistant that provides accurate responses to the multiple-choice trivia questions the user poses.
Results of prompt : +0.2% accuracy"""

base_acc = 955/1409
prompt_num = 1

task_prompts = []
gen_prompts = []
accuracies = []

best_acc = 0.2

for i in range(75):
    task_prompt = f"{SYS_PROMPT}\nOLD PROMPTS\n----------\n{EXAMPLES}\n----------\nUse the information of the performance of the old prompts to generate a modified better prompt. Your new prompt should be relatively different compared to the previous prompts and must not follow the exact wording of any previous prompts.\nNEW PROMPT:"
    task_prompts.append(task_prompt)
    while True:
        try:
            prompt = generate_prompt(task_prompt)
            break
        except:
            continue
    gen_prompts.append(prompt)

    for quantization in quantization_methods:
        quant_path = f"{model_dir}/{quantization}/{model}"
        correct, incorrect, unknown, total = train(trivia_path=trivia_path, 
                                                model_dir=quant_path, 
                                                quantization=quantization, 
                                                remote_path=remote_path, 
                                                cache_dir=cache_dir, 
                                                sys_prompt=prompt,)
        
        
        acc = ((correct / total) - base_acc)*100
        accuracies.append(acc)
        if acc > best_acc:
            prompt_num += 1
            best_acc = acc
            EXAMPLES += f"\n\nPrompt {prompt_num}: {prompt}\nResult of prompt {prompt_num}: +{acc:.1f}% accuracy"

        
    

final_prompt =  f"----------\n{EXAMPLES}\n----------"

with open("output.txt", "w") as f:
    f.write("FINAL PROMPT:\n")
    f.write(final_prompt) 
    f.write("\n------PAST GENERATIONS------\n")
    for iter, tup in enumerate(zip(task_prompts, gen_prompts, accuracies)):
        f.write(f"ITERATION {iter}\n")
        f.write(f"Task prompt:\n{tup[0]}\n\n")
        f.write(f"Gen. prompt: {tup[1]}\n\n")
        f.write(f"Delta accuracy: {tup[2]}\n")
        f.write("----------\n\n")
