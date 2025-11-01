from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from atalaya.terminal import terminal
from atalaya.time_manager import Timer
from atalaya.writer import Writer


def make_synthetic_classification(
    num_samples: int = 4096,
    input_dim: int = 32,
    num_classes: int = 4,
    noise_std: float = 0.2,
) -> DataLoader:
    """Build a toy classification dataset directly in PyTorch."""
    generator = torch.Generator().manual_seed(1234)
    features = torch.randn(num_samples, input_dim, generator=generator)
    prototype_matrix = torch.randn(num_classes, input_dim, generator=generator)
    logits = features @ prototype_matrix.t()
    noise = torch.randn(
        logits.size(),
        generator=generator,
        dtype=logits.dtype,
        device=logits.device,
    )
    logits += noise_std * noise
    targets = torch.argmax(logits, dim=1)

    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=128, shuffle=True)


class MLP(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute accuracy on the provided loader."""
    model.eval()
    num_correct = 0
    num_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            num_correct += (predictions == targets).sum().item()
            num_examples += targets.size(0)
    model.train()
    return num_correct / max(1, num_examples)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_synthetic_classification()
    eval_loader = make_synthetic_classification(num_samples=1024)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    log_dir = Path(f"example/logs/")
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = Writer(
        name="mlp_example",
        project="atalaya_examples",
        logdir=str(log_dir),
        add_name_to_logdir=True,
        add_time=True,
        save_as_csv=True,
        output_catcher=True,
    )
    timer = Timer("mlp_training")

    try:
        with timer:
            global_step = 0
            terminal.print_info("Starting training loop on", device)
            for epoch in range(10):
                running_loss = 0.0
                for batch_inputs, batch_targets in train_loader:
                    optimizer.zero_grad()
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)

                    logits = model(batch_inputs)
                    loss = criterion(logits, batch_targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    # Tensorboard-style logging
                    writer.add_scalars(
                        {"loss": loss.item()},
                        global_step=global_step,
                        prefix="train/batch",
                    )
                    global_step += 1

                avg_loss = running_loss / max(1, len(train_loader))
                train_acc = evaluate(model, train_loader, device)
                eval_acc = evaluate(model, eval_loader, device)

                # Wandb-style logging
                writer.log(
                    {
                        "loss": avg_loss,
                        "train_accuracy": train_acc,
                        "eval_accuracy": eval_acc,
                    },
                    step=epoch,
                    prefix="train",
                )

                terminal.print_info(
                    f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
                    f"train_acc={train_acc:.3f} | eval_acc={eval_acc:.3f}"
                )

            writer.add_models({"mlp": model}, global_step=global_step)
    finally:
        writer.close()
        timer.report(report_type="total_with_stats")
        terminal.print_ok("Example run finished. Logs stored in", log_dir)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        terminal.print_warning("CUDA not detected, falling back to CPU.")
    train()
