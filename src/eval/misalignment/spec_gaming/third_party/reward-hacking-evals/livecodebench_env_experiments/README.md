# LiveCodeBench Reward Hacking Eval

Evaluates whether models engage in reward hacking (hard-coding test cases) on LiveCodeBench coding problems.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:
```
OPENROUTER_API_KEY=your_key_here
```

## (Optional) Regenerate Dataset

Generate the dataset with intentionally incorrect unit tests:
```bash
python create_unit_test_error_dataset.py
```

## Running Evaluations

Run sweeps using:
```bash
python eval_sweep.py
```

Configure hyperparameters in `eval_sweep.py` before running.

See inspect docs for further customization: https://inspect.aisi.org.uk/