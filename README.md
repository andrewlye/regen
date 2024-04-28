# Regenerate: Effective Prompt Optimization for Quantized Language Models
This repository contains the code used in the paper [Regenerate: Effective Prompt Optimization for Quantized Language Models]().

## Quantization
```shell
cd regen

# AWQ
quantization/run-awq.py --model {model path} --bits 4

# GPTQ
quantization/run-gptq.py --model {model path} --bits 4
```

## Prompt Optimization

## Evaluation
