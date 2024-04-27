from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import argparse
import os
os.environ['HOME'] = '/scratch/zx22/andrew'

parser = argparse.ArgumentParser(description='AWQ Quantization')
parser.add_argument('--model_name', default="meta-llama/Llama-2-13b-chat-hf", help='model_name')
parser.add_argument('--cache_dir', default="/home/aly37/.cache", help='cache dir')
parser.add_argument('--save_dir', default="/mnt/vstor/CSE_CSDS_VXC204/aly37/quantize-hal-project/quant-models", help='save dir')
parser.add_argument('--bits', type=int, required=True, help='bits')
parser.add_argument('--token', default='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')
args = parser.parse_args()
print(args)

model_path = args.model_name
quant_path = f'{args.save_dir}/awq/Llama-2-13b-chat-hf'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": args.bits, "version": "GEMM" }

model = AutoAWQForCausalLM.from_pretrained(model_path, cache_dir=args.cache_dir, token=args.token)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=args.token)

model.quantize(tokenizer, quant_config=quant_config)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
