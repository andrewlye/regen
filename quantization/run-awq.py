from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import argparse
import os

parser = argparse.ArgumentParser(description='AWQ Quantization')
parser.add_argument('--model', default="meta-llama/Llama-2-13b-chat-hf", help='model_name')
parser.add_argument('--save', default="quant-models", help='save dir')
parser.add_argument('--bits', type=int, required=True, help='bits')
parser.add_argument('--token', default='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')
args = parser.parse_args()
print(args)

model_path = args.model
quant_path = f'{args.save}/awq/{args.model}'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": args.bits, "version": "GEMM" }

model = AutoAWQForCausalLM.from_pretrained(model_path, device_map='auto', token=args.token)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=args.token)

model.quantize(tokenizer, quant_config=quant_config)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
