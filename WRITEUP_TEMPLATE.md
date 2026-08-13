# Practice Takehome Writeup

## Summary

State your main conclusion in 2-4 sentences.

## Environment

- GPU:
- Python:
- PyTorch:
- Model used for LM probing:
- Total time spent:

## Part A: MNIST Distillation

### Question

What exact question did you test?

### Methods

Describe:

- Teacher architecture and training setup
- Student architecture and training setup
- Label budgets
- Distillation temperature and alpha
- Seeds
- Metrics

### Results

Include a table like:

| Method | Label budget | Temperature | Seed count | Accuracy | NLL | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard | 100 | n/a | 3 | | | |
| kd | 100 | 4 | 3 | | | |

### Interpretation

What changed and why do you think it changed?

### Caveats

List possible confounds:

- Teacher trained on more labels than the student.
- Hyperparameters were not fully tuned.
- MNIST is simple and may not generalize.
- Seed count may be too low.

## Part B: Local LM Probe

### Question

What exact question did you test?

### Methods

Describe:

- Model
- Prompt format
- How answer choices were scored
- What activation or hidden-state measurements were recorded

### Results

Include a table or plot summary:

| Metric | Correct examples | Incorrect examples | Difference |
| --- | ---: | ---: | ---: |
| Mean option margin | | | |
| Mean entropy | | | |
| Mean hidden norm | | | |

### Interpretation

Did confidence-like quantities predict correctness? How much should we trust the pattern?

### Caveats

List possible confounds:

- Tiny prompt set.
- Multiple-choice formatting artifacts.
- Tokenization artifacts for option scoring.
- Hidden norm is a crude activation summary.

## Reproduction

Paste the exact commands you ran.

```bash

```

## Next Experiments

List the highest-value experiments you would run next.

