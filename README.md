# Anthropic ML Takehome Practice

## Overview

Machine-learning takehome practice project using PyTorch and Transformers, with source code, requirements, and writeup template for model experimentation.

## Tech Stack

Python, PyTorch, Transformers, datasets, NumPy, Pandas, Matplotlib, scikit-learn

## Repository Contents

This repository contains the source code and related files for the project. Existing project documentation, submissions, scripts, source folders, data/config files, and supporting assets are preserved below.

## Notes

- Canonical public repository: https://github.com/Shrutibrahma/anthropic-ml-takehome-practice

- README rewritten during portfolio cleanup for clearer presentation.

---

## Original README

# anthropic takehome practice

## Overview

Education analytics / prediction project.

## Tech Stack

Python

## Project Contents

This repository contains the source code and related files for this project as recovered from the local project archive. It may include coursework, prototypes, demos, notebooks, dashboards, firmware/app code, assets, and supporting documentation depending on the original folder.

## Repository Notes

- Original local path: `C:\Users\shrut\Downloads\Shruti_Exam\anthropic_takehome_practice`
- Portfolio category: Takehome / Coding Challenge
- Published repo name: `anthropic-takehome-practice`
- Generated README created during portfolio repository archival.

## How To Use

Inspect the project files for framework-specific setup. Common entry points include README instructions, package manifests, notebooks, Python scripts, app folders, dashboard folders, or firmware/toolchain project files.

---

## Original README

# Practice Takehome: Distillation and Logit Probing

Timebox: 5 hours.

This is a practice project modeled on the skills listed in your email. It is not the real Anthropic task and does not contain inside information. The goal is to practice experimental velocity, conceptual clarity, correctness, and reasoning transparency.

## Scenario

You are given two small research questions.

1. MNIST distillation: Does knowledge distillation from a larger MLP teacher improve a smaller MLP student, especially when the student has limited labeled data?
2. Local language model probing: Can simple logprob and activation-derived measurements predict when a small local language model will choose the correct multiple-choice answer?

You should produce:

- Final code
- A short writeup of results
- Commands needed to reproduce your experiments
- Figures or tables supporting your claims

## Setup

Use a GPU if possible. On Vast.ai, use a PyTorch/CUDA image and run these commands in the remote Linux terminal.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python src/quick_cuda_check.py
```

On Windows PowerShell:

```powershell
uv venv --python 3.11
.\.venv\Scripts\activate
uv pip install -r requirements.txt
python src\quick_cuda_check.py
```

If the PyTorch install in `requirements.txt` does not pick up CUDA on your machine, use the official selector:

https://pytorch.org/get-started/locally/

## Part A: MNIST Distillation

Research question:

Does a larger MLP teacher help a smaller MLP student more than ordinary hard-label training?

Minimum experiments:

- Train a teacher on full MNIST.
- Train a student with hard labels only using label budgets: 100, 1000, full training set.
- Train a student with distillation using the same label budgets.
- Compare at least 2 temperatures from: 1, 2, 4, 8.
- Run at least 3 seeds for the most important comparison if time allows.

Suggested commands:

```bash
python src/mnist_distill.py --method teacher --epochs 5 --seed 0

python src/mnist_distill.py --method hard --label-budget 100 --epochs 5 --seed 0
python src/mnist_distill.py --method hard --label-budget 1000 --epochs 5 --seed 0

python src/mnist_distill.py --method kd --label-budget 100 --temperature 4 --epochs 5 --seed 0 --teacher-ckpt results/mnist_teacher_seed0.pt
python src/mnist_distill.py --method pseudo --label-budget 100 --epochs 5 --seed 0 --teacher-ckpt results/mnist_teacher_seed0.pt
```

Metrics to report:

- Test accuracy
- Test negative log likelihood
- Expected calibration error
- Mean and standard deviation over seeds for your central comparison

Important caveat to discuss:

If the teacher is trained on all MNIST labels, then distillation transfers information from a stronger/full-data model. That is fine, but do not claim the student learned from only 100 labels in the same sense as the hard-label baseline.

## Part B: Local LM Logprob Probe

Research question:

For a small local instruction model, do option logprob margins or hidden-state norms help predict correctness?

Minimum experiments:

- Run a 1B-2B model on the included multiple-choice prompts.
- Score answer choices by logprob rather than only generated text.
- Record option probabilities, entropy, winning margin, correctness, and final-token hidden-state norm.
- Make one plot or table showing whether margin/entropy relates to correctness.

Suggested command:

```bash
python src/llm_probe.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/llm_probe.csv
```

If that model is too slow, use a smaller compatible causal language model.

## Writeup Requirements

Use `WRITEUP_TEMPLATE.md`.

Your writeup should answer:

- What did you test?
- What were the baselines?
- What were the results?
- What uncertainty exists?
- What confounds or failure modes could make the conclusions misleading?
- What would you do next with more time?

## Grading Rubric

Experimental velocity:

- Did you get useful evidence quickly?
- Did you avoid getting stuck on polish?

Conceptual understanding:

- Do you understand what distillation changes?
- Do you understand logits, logprobs, entropy, and activations?

Correctness and validity:

- No train/test leakage.
- Baselines are fair enough for the claim being made.
- Commands are reproducible.

Reasoning transparency:

- Do not overstate small or noisy differences.
- Report seeds, error bars, or at least clear uncertainty.
- Name confounds plainly.



