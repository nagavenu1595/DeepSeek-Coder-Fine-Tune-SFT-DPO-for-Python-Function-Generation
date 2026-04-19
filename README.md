# DeepSeek-Coder Fine-Tune: SFT + DPO for Python Function Generation

A student research project fine-tuning DeepSeek-Coder 1.3B on a custom dataset using Supervised Fine-Tuning and Direct Preference Optimization to generate simple Python functions from natural language task descriptions. Trained entirely on a free Kaggle GPU.

---

## Honest Summary

This project successfully demonstrates a complete end-to-end LLM fine-tuning pipeline. The fine-tuned model performs well on simple, common tasks (palindrome, vowel counting, finding max) but struggles with tasks requiring multi-step logic (factorial, prime checking) due to the small dataset size and limited model capacity. The base model (DeepSeek-Coder 1.3B without fine-tuning) still outperforms the fine-tuned version on general, unseen tasks — which is a known limitation of fine-tuning on small, narrow datasets.

---

## Project Structure

```
├── final_code_buddy_v5_clean.ipynb   # Training notebook (SFT + DPO pipeline)
├── infer.py                          # CLI inference script
├── grpo_training_data_v5.json        # Training dataset (115 examples)
└── README.md
```

---

## How It Works

### Input → Output
```
"write a function to check if a string is a palindrome"
                        ↓
        def is_palindrome(s):
            return s == s[::-1]
```

### Training Pipeline
```
Dataset (115 examples)
        ↓
  Reward Scoring        ← execution-based, runs assert test cases
        ↓
  Stage 1: SFT          ← train on high-quality completions only (reward ≥ 0.9)
        ↓
  Stage 2: DPO          ← learn to prefer correct code over buggy/stub code
        ↓
  Saved LoRA Adapter    ← ~few MB, loads on top of base model
        ↓
  infer.py              ← interactive CLI tool
```

---

## Model & Training Details

| Component | Value |
|---|---|
| Base model | deepseek-ai/DeepSeek-Coder-1.3B-instruct |
| Parameters trained | ~1-2% (LoRA adapters only) |
| Quantization | 4-bit NF4 (T4/A100) or 8-bit (P100) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| DPO beta | 1.5 |
| DPO learning rate | 2e-5 |
| DPO epochs | 1 |
| Training hardware | Kaggle free tier (T4 / P100) |
| Dataset size | 115 examples |

---

## Dataset

Custom dataset (`grpo_training_data_v5.json`) with 115 Python coding tasks. Each entry contains:

- `prompt` — natural language task description
- `completion` — a correct Python function
- `test_cases` — assert-based unit tests for execution scoring
- `graded_rejected` — buggy or stub versions used as DPO negatives

### Reward Function

The reward function executes generated code against test cases and scores it:

| Condition | Score |
|---|---|
| Syntax error | 0.05 |
| No function definition | 0.10 |
| Tests fail entirely | 0.10 |
| Partial pass rate p | 0.10 + p × 0.90 |
| All tests pass | 1.00 |

---

## Evaluation Results

Evaluated on 5 held-out tasks after DPO training:

| Task | Score | Status |
|---|---|---|
| Check palindrome | 1.00 | ✅ Pass |
| Find maximum in list | 1.00 | ✅ Pass |
| Count vowels | 1.00 | ✅ Pass |
| Compute factorial | 0.10 | ❌ Fail |
| Check prime number | 0.05 | ❌ Fail |

**Mean reward: 0.630 — Pass rate: 3/5 (60%)**

---

## Running Inference

```bash
pip install torch transformers peft
python infer.py
```

```
=== Code Generator | type 'exit' to quit ===

Task: write a function to reverse a string

Generating...

```python
def reverse_string(s):
    return s[::-1]
```
```

---

## Requirements

```
torch
transformers==4.44.2
peft==0.12.0
accelerate==0.34.2
datasets>=2.20.0
trl==0.9.6
bitsandbytes>=0.43.0
safetensors
numpy
scipy
matplotlib
psutil
```

---

## Known Limitations

- **Small dataset** — 115 examples is not enough for reliable generalization. The model defaults to familiar patterns on unseen tasks.
- **Incomplete generation** — for multi-step logic functions (factorial, prime), the model sometimes generates only a docstring or partial body before stopping.
- **No benchmark evaluation** — not tested on standard benchmarks like HumanEval or MBPP.
- **Base model regression** — the fine-tuned model underperforms the base model on general tasks outside the training distribution. This is a known effect of fine-tuning on small, narrow datasets (alignment tax).
- **Simple tasks only** — works reliably for single-return-statement functions. Complex multi-step logic, recursion, and nested conditions are inconsistent.

---

## Future Work

- Scale dataset to 500+ examples with diverse task types
- Add more test cases per example (currently 3) to make reward scoring more meaningful
- Evaluate on HumanEval and MBPP benchmarks
- Try larger model: DeepSeek-Coder 6.7B
- Add GRPO stage for online reinforcement from code execution
- Build a Gradio or VS Code extension interface

---

## Author

Nagarajan Venugopal — 23BRS1060
