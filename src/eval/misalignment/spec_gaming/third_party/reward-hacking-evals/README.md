# reward-hack-evals

This repository contains a number of environments evaluating model reward hacking or evaluation gaming propensities. This repo will be cleaned up over the next few weeks.

## Eval Locations

The evaluation environments are organized across the following locations:

- **Customer Service** - `runner.py`
- **Email Assistant** - `runner.py`
- **Multiple Choice** - `runner.py`
- **Data Entry** - `inspect_command_line_envs/cl_agent_envs.py`
- **Sales** - `sales_env_experiments/`
- **LiveCode** - `livecodebench_env_experiments/`

## Source Datasets

Some of the evaluations in this repository leverage pre-existing datasets. These datasets are:

- [MMLU - Massive Multitask Language Understanding](https://github.com/hendrycks/test/tree/master) - [(MIT License)](https://github.com/hendrycks/test/blob/master/LICENSE)

- [Measuring Coding Challenge Competence With APPS](https://github.com/hendrycks/apps) - [(MIT License)](https://github.com/hendrycks/apps/blob/main/LICENSE)

- [Measuring Mathematical Problem Solving With the MATH Dataset](https://github.com/hendrycks/math) - [(MIT License)](https://github.com/hendrycks/math/blob/main/LICENSE)

- [Extreme Summarization](https://github.com/EdinburghNLP/XSum) - [(MIT License)](https://github.com/EdinburghNLP/XSum/blob/master/LICENSE)

- [LiveCodeBench - Holistic and Contamination Free Evaluation of LLMs for Code](https://github.com/LiveCodeBench/LiveCodeBench) - [(MIT License)](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/LICENSE)
