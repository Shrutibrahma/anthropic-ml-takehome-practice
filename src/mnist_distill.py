from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from metrics import classification_metrics, set_seed
from models import MLP


class IndexedSubset(Dataset):
    def __init__(self, base: Dataset, indices: list[int] | None = None) -> None:
        self.base = base
        self.indices = list(range(len(base))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        x, y = self.base[idx]
        return x, y, idx


def balanced_label_indices(targets: list[int], budget: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets)
    if budget >= len(labels):
        return list(range(len(labels)))

    per_class = budget // 10
    remainder = budget % 10
    chosen: list[int] = []
    for cls in range(10):
        cls_indices = np.flatnonzero(labels == cls)
        n = per_class + (1 if cls < remainder else 0)
        chosen.extend(rng.choice(cls_indices, size=n, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def get_data(data_dir: Path, batch_size: int, label_budget: int, seed: int):
    transform = transforms.ToTensor()
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    label_indices = balanced_label_indices(train.targets, label_budget, seed)
    label_set = set(label_indices)

    hard_loader = DataLoader(
        IndexedSubset(train, label_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    full_train_loader = DataLoader(
        IndexedSubset(train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        IndexedSubset(test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return hard_loader, full_train_loader, test_loader, label_set


def build_model(kind: str, args: argparse.Namespace) -> MLP:
    if kind == "teacher":
        return MLP(hidden_dim=args.teacher_hidden, depth=args.teacher_depth, dropout=args.dropout)
    return MLP(hidden_dim=args.student_hidden, depth=args.student_depth, dropout=args.dropout)


def train_epoch(
    model: MLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    method: str,
    label_set: set[int],
    teacher: MLP | None,
    temperature: float,
    alpha: float,
) -> float:
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_seen = 0

    for x, y, idx in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        idx_np = idx.numpy()
        logits = model(x)

        if method in {"teacher", "hard", "supervised_all"}:
            loss = F.cross_entropy(logits, y)
        else:
            assert teacher is not None
            with torch.no_grad():
                teacher_logits = teacher(x)

            if method == "pseudo":
                pseudo = teacher_logits.argmax(dim=-1)
                loss = F.cross_entropy(logits, pseudo)
            elif method == "kd":
                student_log_probs = F.log_softmax(logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature * temperature

                labeled_mask_np = np.asarray([int(i) in label_set for i in idx_np])
                labeled_mask = torch.as_tensor(labeled_mask_np, dtype=torch.bool, device=device)
                if labeled_mask.any():
                    hard_loss = F.cross_entropy(logits[labeled_mask], y[labeled_mask])
                    loss = alpha * hard_loss + (1.0 - alpha) * kd_loss
                else:
                    loss = kd_loss
            else:
                raise ValueError(f"Unknown method: {method}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_seen += x.size(0)

    return total_loss / max(total_seen, 1)


@torch.no_grad()
def evaluate(model: MLP, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []
    for x, y, _idx in loader:
        x = x.to(device)
        y = y.to(device)
        all_logits.append(model(x))
        all_labels.append(y)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return classification_metrics(logits, labels)


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["teacher", "hard", "supervised_all", "pseudo", "kd"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--teacher-ckpt", type=Path, default=None)
    parser.add_argument("--label-budget", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight on hard-label loss in KD.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-hidden", type=int, default=512)
    parser.add_argument("--teacher-depth", type=int, default=3)
    parser.add_argument("--student-hidden", type=int, default=128)
    parser.add_argument("--student-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    hard_loader, full_train_loader, test_loader, label_set = get_data(
        args.data_dir, args.batch_size, args.label_budget, args.seed
    )

    model_kind = "teacher" if args.method == "teacher" else "student"
    model = build_model(model_kind, args).to(device)

    teacher = None
    if args.method in {"pseudo", "kd"}:
        if args.teacher_ckpt is None:
            raise ValueError("pseudo and kd require --teacher-ckpt")
        teacher = build_model("teacher", args).to(device)
        checkpoint = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(checkpoint["model_state"])

    if args.method == "hard":
        train_loader = hard_loader
    else:
        train_loader = full_train_loader

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.method,
            label_set,
            teacher,
            args.temperature,
            args.alpha,
        )
        metrics = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch + 1} loss={train_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} nll={metrics['nll']:.4f} ece={metrics['ece']:.4f}"
        )

    final_metrics = evaluate(model, test_loader, device)

    if args.method == "teacher":
        ckpt_path = args.results_dir / f"mnist_teacher_seed{args.seed}.pt"
    else:
        ckpt_path = args.results_dir / (
            f"mnist_{args.method}_budget{args.label_budget}_temp{args.temperature}_seed{args.seed}.pt"
        )
    torch.save({"model_state": model.state_dict(), "args": vars(args), "metrics": final_metrics}, ckpt_path)

    row = {
        "method": args.method,
        "label_budget": args.label_budget,
        "temperature": args.temperature if args.method == "kd" else "",
        "alpha": args.alpha if args.method == "kd" else "",
        "seed": args.seed,
        "epochs": args.epochs,
        "accuracy": final_metrics["accuracy"],
        "nll": final_metrics["nll"],
        "ece": final_metrics["ece"],
        "checkpoint": str(ckpt_path),
    }
    append_result(args.results_dir / "mnist_results.csv", row)
    print(f"saved: {ckpt_path}")
    print(f"logged: {args.results_dir / 'mnist_results.csv'}")


if __name__ == "__main__":
    main()

