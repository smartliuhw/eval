# Intro

This repo is used for @smartliuhw thesis's model evaluation. The [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is used as the basic framework.

The code was running on the RTX 4090 with 24G GPU-memory with [accelarate](https://github.com/huggingface/accelerate) package to enable data parallel.

# How to use

## Install the dependency

Enter the path which contanins the README file, then run the following command:

```bash
pip install -e .
```

## Modify task configurations

All the task configurations are in this [path](./lm_eval/tasks/). Enter it and modify the task's configuration you need.

An example is in [nq_open_cot.yaml](./lm_eval/tasks/nq_open/nq_open_cot.yaml) file, in which I customized the dataset path, task group, descriptions, input template and metrics. Also, the [utils.py](./lm_eval/tasks/nq_open/utils.py) is modified to adapt to the special dataset. It is recommanded to save the dataset locally to save precious time.

## Modify evaluation script

After customized the task you need, a shell script is needed to launch the evaluation. An example is in [eval_test.sh](./eval_test.sh) file. Only a few params are needed to be changed.

If you have any question, feel free to ask me. And it's recommanded to read [the framework's origin README](./README_Repo.md) to gain a better understanding about this framework.