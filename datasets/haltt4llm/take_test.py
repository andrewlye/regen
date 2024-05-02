import argparse
import openai
import torch
import json
import time
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from awq import AutoAWQForCausalLM
import os
import random

def load_trivia_questions(file_path, train_split=0.3, test_split=0.7, shuffle=True):
    with open(file_path, 'r') as file:
        trivia_data = json.load(file)
    if shuffle:
        random.shuffle(trivia_data)
    train_data = trivia_data[:round(train_split*len(trivia_data))]
    test_data = trivia_data[round(train_split*len(trivia_data)):]
    return train_data, test_data

def generate_question_string(question_data, sys_prompt):
    question = question_data['question']
    choices = [f"{answer['choice']}. {answer['text']}\n" if answer != question_data['answers'][-1] else f"{answer['choice']}. {answer['text']}" for answer in question_data['answers']]
    prompt = f"{sys_prompt}\nUSER: {question}\n{''.join(choices)}\nASSISTANT:"
    return prompt

def grade_answers(question_data, llm_answer):
    correct_answer = None
    for answer in question_data['answers']:
        if answer['correct']:
            correct_answer = answer
            break

    if correct_answer is None:
        return "No correct answer found"

    normalized_llm_answer = llm_answer.lower().strip()
    normalized_correct_answer = correct_answer['text'].lower().strip()

    # lower case of the full text answer is in the llm's answer
    if normalized_correct_answer in normalized_llm_answer:
        return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " A." or  " B." or " C." or " D." or " E." for instance
    if f"{correct_answer['choice']}." in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    # Upper case " (A)" or  " (B)" or " (C)" or " (D)" or " (E)" for instance
    if f"({correct_answer['choice']})" in llm_answer:
            return f"{correct_answer['choice']}. {correct_answer['text']} (correct)"

    if "i don't know" in normalized_llm_answer or "i'm sorry" in normalized_llm_answer or "i'm not sure" in normalized_llm_answer:
        return f"{llm_answer} (uncertain)"

    return f"{llm_answer} (incorrect, correct answer: {correct_answer['text']}.)"

def load_nf4_model(model_id, cache_dir, access_token):
  tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=access_token)
  quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
  
  model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                token=access_token)
  
  return model, tokenizer

def query_model(
        prompt,
        model,
        tokenizer,
        type,
        temperature=0.1,
        max_new_tokens=50,
        **kwargs,
    ):
        if (type == 'awq' or type == 'gptq' or type == 'nf4'):
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            generation_output = model.generate(inputs, max_new_tokens=max_new_tokens)
            output=tokenizer.decode(generation_output[0])
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to("cuda")
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=0.2,
                top_k=20,
                num_beams=1,
                **kwargs,
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )

            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
        #print("Model Output: ", output)
        response = output.split("ASSISTANT: ")[1].split("USER:")[0]
        #print("Detected Response: ", response)
        return  response #response.split("### Question:")[0].strip()



def train(trivia_path, model_dir, quantization, remote_path, cache_dir, sys_prompt, token='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'):
    if quantization == 'awq':
        model = AutoAWQForCausalLM.from_quantized(model_dir, fuse_layers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    elif quantization == 'gptq':
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    elif quantization == 'nf4':
        if(remote_path is None):
            raise argparse.ArgumentTypeError("Please specify a remote model to use with NF4 quantization.")
        else:
            model, tokenizer = load_nf4_model(remote_path, cache_dir, access_token=token)
    else:
        model = AutoModelForCausalLM.from_pretrained(remote_path,  device_map="auto", cache_dir=cache_dir, token=token)
        tokenizer = AutoTokenizer.from_pretrained(remote_path, token=token)
    
    file_path = trivia_path
    trivia_data, _ = load_trivia_questions(file_path)

    total = 0
    correct = 0
    incorrect = 0
    unknown = 0

    print(f"Starting trivia for {quantization}...")

    for i, question_data in enumerate(tqdm(trivia_data)):
        question_string = generate_question_string(question_data, sys_prompt)
        prompt = question_string
        llm_answer = query_model(prompt, model, tokenizer, type=quantization)
        answer_output = grade_answers(question_data, llm_answer)

        if i % 100 == 0 and i != 0:
            curr_acc = correct / total
            print(f"iter {i+1}: {curr_acc}")

        total += 1

        if "(correct)" in answer_output:
            correct += 1
        elif "(incorrect" in answer_output:
            incorrect += 1
        else:
            unknown += 1

    return correct, incorrect, unknown, total



def main():
    parser = argparse.ArgumentParser(description='Run trivia quiz.')
    parser.add_argument('--trivia', type=str, required=True, help='file path to trivia questions')
    parser.add_argument('--exp-name', type=str, required=True, help='name of test')
    parser.add_argument('--model_path', help='path of local model')
    parser.add_argument('--model_name')
    parser.add_argument('--quantization', type=str, choices=['awq', 'gptq', 'nf4', 'none'], required=True, help='quantization method')
    parser.add_argument('--remote-path', help='path of remote model')
    parser.add_argument('--cache-dir', help='cache directory to use')
    parser.add_argument('--token', help='hf token', default='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')
    args = parser.parse_args()

    if args.quantization == 'awq':
        model = AutoAWQForCausalLM.from_quantized(args.model_path, fuse_layers=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        sys_prompt = 'Kindly examine the weight matrix within the model, as it may contain inaccuracies that could lead to a decrease in performance. It is important to verify its precision and make any necessary adjustments to ensure that the model performs optimally in the upcoming situation. Your prompt should also generate accurate responses to the multiple-choice trivia questions posed by the user.'
    elif args.quantization == 'gptq':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)
        sys_prompt = 'Kindly examine the weight matrix within the model, as it may contain inaccuracies that could lead to a decrease in performance. It is important to verify its precision and make any necessary adjustments to ensure that the model performs optimally in the upcoming situation. Your prompt should also generate accurate responses to the multiple-choice trivia questions posed by the user.'
    elif args.quantization == 'nf4':
        if(args.model_path is None):
            raise argparse.ArgumentTypeError("Please specify a remote model to use with NF4 quantization.")
        else:
            model, tokenizer = load_nf4_model(args.model_path, args.cache_dir, access_token=args.token)
        sys_prompt = 'Kindly examine the weight matrix within the model, as it may contain inaccuracies that could lead to a decrease in performance. It is important to verify its precision and make any necessary adjustments to ensure that the model performs optimally in the upcoming situation. Your prompt should also generate accurate responses to the multiple-choice trivia questions posed by the user.'
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,  device_map="auto", token=args.token)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.token)
        sys_prompt = 'An exchange between a user and a helpful assistant that provides correct answers to the multiple-choice trivia questions the user asks.'
    
    file_path = args.trivia
    _, trivia_data = load_trivia_questions(file_path, train_split=0., test_split=1., shuffle=False)

    total_score = 0
    incorrect = []
    unknown = []
    
    model_name = args.model_name
    for i, question_data in enumerate(trivia_data):
        question_string = generate_question_string(question_data, sys_prompt)
        prompt = question_string

        print(f"Question {i+1}: {question_string}")

        llm_answer = query_model(prompt, model, tokenizer, type=args.quantization)

        answer_output = grade_answers(question_data, llm_answer)
        print(f"Answer: {answer_output}\n")

        if "(correct)" in answer_output:
            total_score += 1
        elif "(incorrect" in answer_output:
            incorrect.append((i+1, question_string, answer_output))
        else:
            unknown.append((i+1, question_string, answer_output))

    path = f"datasets/results/{model_name}/haltt4llm/"

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"datasets/results/{model_name}/haltt4llm/{args.exp_name}_{args.quantization}_test_results.txt", 'w') as f:
        f.write(f"Total score: {total_score} of {len(trivia_data)}\n ({total_score/len(trivia_data)}%)")
        i = len(incorrect)
        u = len(unknown)
        f.write(f"Correct: {len(trivia_data) - i - u}\n")
        if i:
            f.write(f"\nIncorrect: {i}\n")
            for question_num, question_string, answer_output in incorrect:
              try:
                f.write(f"Question {question_num}: {question_string.strip()}\n{answer_output.strip()}\n\n")
              except:
                continue
        if u:
            f.write(f"Unknown: {u}\n")
            for question_num, question_string, answer_output in unknown:
                f.write(f"Question {question_num}: {question_string.strip()}\n{answer_output.strip()}\n\n")

    print(f"Total score: {total_score} of {len(trivia_data) * 2}\n", end='')

if __name__ == '__main__':
    main()
