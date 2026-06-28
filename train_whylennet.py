#!/usr/bin/env python3
"""Self-contained training harness for the experimental WhyLeNet architecture.

WhyLeNet is intentionally strange: each macro hidden unit is a "WhyNeuron" that
turns a 10-way activation distribution into a blended MNIST-like image, then asks
small internal LeNets to predict the tens and ones digits of a continuous scalar.
The script trains the full model end-to-end on MNIST without argmax/rounding in
any forward pass path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class EntropyStats:
    """Aggregated inner Mini-LeNet certainty diagnostics."""

    entropy_sum: float = 0.0
    certainty_sum: float = 0.0
    count: int = 0

    def update(self, entropy: Tensor, certainty: Tensor) -> None:
        self.entropy_sum += float(entropy.detach().sum().cpu())
        self.certainty_sum += float(certainty.detach().sum().cpu())
        self.count += int(entropy.numel())

    @property
    def entropy(self) -> float:
        return self.entropy_sum / max(1, self.count)

    @property
    def certainty(self) -> float:
        return self.certainty_sum / max(1, self.count)


class DigitBank(nn.Module):
    """A device-aware bank of representative MNIST digit templates.

    The bank is created from the average image for each class in the MNIST
    training set, has shape [10, 1, 28, 28], and is registered as a buffer so
    `.to(device)` moves it with the rest of the model.
    """

    def __init__(self, mnist_train: datasets.MNIST) -> None:
        super().__init__()
        templates = self._build_class_averages(mnist_train)
        self.register_buffer("templates", templates, persistent=True)

    @staticmethod
    def _build_class_averages(mnist_train: datasets.MNIST) -> Tensor:
        sums = torch.zeros(10, 1, 28, 28, dtype=torch.float32)
        counts = torch.zeros(10, dtype=torch.float32)

        # MNIST.data is uint8 [N, 28, 28]; targets are class ids. This happens
        # once at startup and is not part of the differentiable model forward.
        images = mnist_train.data.float().div(255.0).unsqueeze(1)
        targets = torch.as_tensor(mnist_train.targets, dtype=torch.long)
        for digit in range(10):
            mask = targets == digit
            if not bool(mask.any()):
                raise ValueError(f"MNIST training set contains no examples for digit {digit}")
            sums[digit] = images[mask].mean(dim=0)
            counts[digit] = mask.sum()

        if not torch.all(counts > 0):
            raise ValueError("Could not build all ten digit templates")
        return sums


class MiniLeNetRegressor(nn.Module):
    """Tiny LeNet-style image-to-10-logit module used inside each WhyNeuron."""

    def __init__(self, c1: int = 2, c2: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5),  # 28 -> 24
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),  # 24 -> 12
            nn.Conv2d(c1, c2, kernel_size=5),  # 12 -> 8
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),  # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 4 * 4, 16),
            nn.Tanh(),
            nn.Linear(16, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class WhyNeuron(nn.Module):
    """Base-10 super-neuron that synthesizes an image and emits one scalar.

    Input: [batch, 10] activation logits or probabilities.
    Output: [batch, 1] continuous value in approximately [0, 99].
    """

    def __init__(self, digit_bank: DigitBank) -> None:
        super().__init__()
        self.digit_bank = digit_bank
        self.tens = MiniLeNetRegressor()
        self.ones = MiniLeNetRegressor()
        self.register_buffer("digit_values", torch.arange(10, dtype=torch.float32), persistent=False)

    def forward(self, x: Tensor, collect_stats: bool = False) -> Tensor | Tuple[Tensor, EntropyStats]:
        weights = F.softmax(x, dim=-1)
        blended = torch.einsum("bi,ichw->bchw", weights, self.digit_bank.templates)

        tens_logits = self.tens(blended)
        ones_logits = self.ones(blended)
        tens_probs = F.softmax(tens_logits, dim=-1)
        ones_probs = F.softmax(ones_logits, dim=-1)

        tens_expected = (tens_probs * self.digit_values).sum(dim=-1, keepdim=True)
        ones_expected = (ones_probs * self.digit_values).sum(dim=-1, keepdim=True)
        output = 10.0 * tens_expected + ones_expected

        if not collect_stats:
            return output

        stats = EntropyStats()
        for probs in (tens_probs, ones_probs):
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
            certainty = probs.max(dim=-1).values
            stats.update(entropy, certainty)
        return output, stats


class WhyLinearLayer(nn.Module):
    """Linear-like layer backed by a ModuleList of WhyNeurons.

    Each output neuron receives the same 10-way distribution and emits one scalar,
    producing [batch, out_features].
    """

    def __init__(self, out_features: int, digit_bank: DigitBank) -> None:
        super().__init__()
        self.neurons = nn.ModuleList([WhyNeuron(digit_bank) for _ in range(out_features)])

    def forward(self, x: Tensor, collect_stats: bool = False) -> Tensor | Tuple[Tensor, EntropyStats]:
        outputs: List[Tensor] = []
        merged_stats = EntropyStats()
        for neuron in self.neurons:
            if collect_stats:
                value, stats = neuron(x, collect_stats=True)
                merged_stats.entropy_sum += stats.entropy_sum
                merged_stats.certainty_sum += stats.certainty_sum
                merged_stats.count += stats.count
            else:
                value = neuron(x)
            outputs.append(value)
        stacked = torch.cat(outputs, dim=1)
        return (stacked, merged_stats) if collect_stats else stacked


class WhyLeNet(nn.Module):
    """Compact MNIST classifier using one WhyLinearLayer as its hidden stage."""

    def __init__(self, digit_bank: DigitBank, hidden_neurons: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 10),
        )
        self.why = WhyLinearLayer(hidden_neurons, digit_bank)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_neurons),
            nn.Linear(hidden_neurons, 10),
        )

    def forward(self, x: Tensor, collect_stats: bool = False) -> Tensor | Tuple[Tensor, EntropyStats]:
        distribution_logits = self.encoder(x)
        if collect_stats:
            why_values, stats = self.why(distribution_logits, collect_stats=True)
            return self.classifier(why_values), stats
        why_values = self.why(distribution_logits)
        return self.classifier(why_values)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_subset(dataset: datasets.MNIST, limit: int | None) -> datasets.MNIST | Subset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def train_one_epoch(model: WhyLeNet, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    stats = EntropyStats()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, batch_stats = model(images, collect_stats=True)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total += batch_size
        stats.entropy_sum += batch_stats.entropy_sum
        stats.certainty_sum += batch_stats.certainty_sum
        stats.count += batch_stats.count

    return {"loss": total_loss / total, "accuracy": correct / total, "entropy": stats.entropy, "certainty": stats.certainty}


@torch.no_grad()
def evaluate(model: WhyLeNet, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    stats = EntropyStats()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, batch_stats = model(images, collect_stats=True)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.cpu()) * batch_size
        correct += int((logits.argmax(dim=-1) == labels).sum().cpu())
        total += batch_size
        stats.entropy_sum += batch_stats.entropy_sum
        stats.certainty_sum += batch_stats.certainty_sum
        stats.count += batch_stats.count

    return {"loss": total_loss / total, "accuracy": correct / total, "entropy": stats.entropy, "certainty": stats.certainty}


def build_loaders(data_dir: Path, batch_size: int, train_limit: int | None, test_limit: int | None) -> Tuple[datasets.MNIST, DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_full = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(maybe_subset(train_full, train_limit), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(maybe_subset(test_full, test_limit), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    return train_full, train_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the experimental WhyLeNet architecture on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-neurons", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=0, help="Optional quick-run cap; 0 means full training set.")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional quick-run cap; 0 means full test set.")
    parser.add_argument("--checkpoint", type=Path, default=Path("whylennet.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    device = select_device()
    print(f"Using device: {device}")

    train_full, train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.train_limit or None, args.test_limit or None)
    digit_bank = DigitBank(train_full)
    model = WhyLeNet(digit_bank=digit_bank, hidden_neurons=args.hidden_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3%} "
            f"entropy {train_metrics['entropy']:.3f} certainty {train_metrics['certainty']:.3f} | "
            f"test loss {eval_metrics['loss']:.4f} acc {eval_metrics['accuracy']:.3%} "
            f"entropy {eval_metrics['entropy']:.3f} certainty {eval_metrics['certainty']:.3f}"
        )

    checkpoint = {"model_state": model.state_dict(), "args": vars(args)}
    torch.save(checkpoint, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
