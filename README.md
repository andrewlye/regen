# [Working Title] Regenerate: Effective Prompt Optimization for Quantized Language Models
This repository contains the code used in the paper [Regenerate: Effective Prompt Optimization for Quantized Language Models]().

# Tutorial
```shell
cd regen
```
## Quantization
```shell
# AWQ
quantization/run-awq.py --model {model path} --bits 4

# GPTQ
quantization/run-gptq.py --model {model path} --bits 4
```

## Prompt Optimization
```shell
python prompt_learn.py
```

## Evaluation
```shell
# TruthfulQA
datasets/trustllmr/truthfulness/take_test.py 
```
