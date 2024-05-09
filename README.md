# [Working Title] Regenerate: Effective Prompt Optimization for Quantized Language Models
This repository contains the code used in the paper [PLACEHOLDER].

# Tutorial
```shell
cd regen
```
## Quantization
```shell
# AWQ
python quantization/run-awq.py --model {model path} --bits 4

# GPTQ
python quantization/run-gptq.py --model {model path} --bits 4
```

## Prompt Optimization
For now, open prompt_learn.py and manually change lines 31-40.
```shell
python prompt_learn.py
```

## Evaluation
Check `datasets/trustllm/truthfulness/take_test.py` for argparse params.
```shell
# TruthfulQA
python datasets/trustllm/truthfulness/take_test.py --args
```
