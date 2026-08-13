from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


TASKS = [
    {
        "id": "arithmetic_1",
        "question": "What is 17 + 28?",
        "choices": {"A": "35", "B": "45", "C": "54"},
        "answer": "B",
    },
    {
        "id": "arithmetic_2",
        "question": "A box has 6 red balls and 4 blue balls. If 3 red balls are removed, how many red balls remain?",
        "choices": {"A": "3", "B": "4", "C": "7"},
        "answer": "A",
    },
    {
        "id": "logic_1",
        "question": "All blickets are daxes. No daxes are wugs. Can a blicket be a wug?",
        "choices": {"A": "Yes", "B": "No", "C": "Cannot tell"},
        "answer": "B",
    },
    {
        "id": "sentiment_1",
        "question": "The review says: 'The food arrived cold, but the staff fixed the issue quickly.' What is the overall sentiment?",
        "choices": {"A": "Mixed", "B": "Very negative", "C": "Unrelated"},
        "answer": "A",
    },
    {
        "id": "world_1",
        "question": "Which planet is known for prominent rings visible from Earth-based telescopes?",
        "choices": {"A": "Mars", "B": "Saturn", "C": "Mercury"},
        "answer": "B",
    },
    {
        "id": "trap_1",
        "question": "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "choices": {"A": "$0.05", "B": "$0.10", "C": "$1.00"},
        "answer": "A",
    },
]


def make_prompt(task: dict) -> str:
    choices = "\n".join(f"{key}. {value}" for key, value in task["choices"].items())
    return (
        "Answer the multiple-choice question. Give only the letter.\n\n"
        f"Question: {task['question']}\n"
        f"Choices:\n{choices}\n"
        "Answer:"
    )


@torch.no_grad()
def completion_logprob(model, tokenizer, prompt: str, completion: str, device: torch.device) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + completion, add_special_tokens=False).input_ids
    if len(full_ids) <= len(prompt_ids):
        return float("-inf")

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids).logits[0]
    log_probs = F.log_softmax(logits[:-1], dim=-1)

    total = 0.0
    for token_pos in range(len(prompt_ids), len(full_ids)):
        target_id = full_ids[token_pos]
        total += float(log_probs[token_pos - 1, target_id].detach().cpu())
    return total


@torch.no_grad()
def final_token_hidden_norm(model, tokenizer, prompt: str, device: torch.device) -> float:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    hidden = out.hidden_states[-1][0, -1].float()
    return float(hidden.norm().detach().cpu())


def normalized_option_stats(scores: dict[str, float]) -> dict[str, float | str]:
    labels = list(scores)
    values = torch.tensor([scores[k] for k in labels], dtype=torch.float64)
    probs = torch.softmax(values, dim=0)
    order = torch.argsort(probs, descending=True)
    pred = labels[int(order[0])]
    margin = float(probs[order[0]] - probs[order[1]]) if len(labels) > 1 else 1.0
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum())
    out: dict[str, float | str] = {"pred": pred, "margin": margin, "entropy": entropy}
    for label, prob in zip(labels, probs):
        out[f"prob_{label}"] = float(prob)
        out[f"logprob_{label}"] = scores[label]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("results/llm_probe.csv"))
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    model.eval()

    rows = []
    for task in TASKS:
        prompt = make_prompt(task)
        scores = {}
        for label in task["choices"]:
            scores[label] = completion_logprob(model, tokenizer, prompt, " " + label, device)

        stats = normalized_option_stats(scores)
        hidden_norm = final_token_hidden_norm(model, tokenizer, prompt, device)
        row = {
            "id": task["id"],
            "answer": task["answer"],
            "pred": stats["pred"],
            "correct": stats["pred"] == task["answer"],
            "margin": stats["margin"],
            "entropy": stats["entropy"],
            "hidden_norm": hidden_norm,
        }
        for key, value in stats.items():
            if key not in row:
                row[key] = value
        rows.append(row)
        print(
            f"{task['id']}: pred={row['pred']} answer={row['answer']} "
            f"correct={row['correct']} margin={row['margin']:.3f} entropy={row['entropy']:.3f}"
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    acc = df["correct"].mean()
    correct = df[df["correct"]]
    wrong = df[~df["correct"]]
    print(f"\naccuracy={acc:.3f} n={len(df)}")
    if len(correct) > 0 and len(wrong) > 0:
        print(f"mean_margin_correct={correct['margin'].mean():.3f}")
        print(f"mean_margin_wrong={wrong['margin'].mean():.3f}")
        print(f"mean_entropy_correct={correct['entropy'].mean():.3f}")
        print(f"mean_entropy_wrong={wrong['entropy'].mean():.3f}")
    else:
        print("Need both correct and incorrect examples to compare confidence statistics.")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

