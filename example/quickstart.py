"""The atl.init / atl.log way, on a toy regression.

    python example/quickstart.py
    tensorboard --logdir example_logs
"""

import random

import atalaya as atl


@atl.timer("step")  # decorating before atl.init() is fine
def train_step(w, lr):
    x = random.uniform(-1, 1)
    y = 3.0 * x + random.gauss(0, 0.1)
    grad = 2 * (w * x - y) * x
    return w - lr * grad, (w * x - y) ** 2


def main():
    atl.init(project="quickstart", logdir="example_logs", config={"lr": 0.05, "epochs": 20})
    w = 0.0
    for epoch in range(atl.config["epochs"]):
        losses = []
        for _ in range(100):
            w, loss = train_step(w, atl.config["lr"])
            losses.append(loss)
        atl.log({"loss": sum(losses) / len(losses), "w": w})
        if epoch % 5 == 0:
            atl.log_timings()
    atl.terminal.print_ok(f"w = {w:.3f} (true 3.0); logs in {atl.run.logdir}")
    atl.finish()


if __name__ == "__main__":
    main()
