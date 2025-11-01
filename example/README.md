# Atalaya Example: MLP Training

This folder contains a minimal PyTorch example that demonstrates how to use `atalaya.Writer`, the terminal helper, and the timer utilities inside a toy training loop.

## Prerequisites

- Python 3.10+
- PyTorch (install following the [official instructions](https://pytorch.org/get-started/locally/))
- Atalaya (`pip install atalaya`)

```bash
python example/mlp.py
```

The script trains a small multi-layer perceptron on synthetic data for a few epochs while logging batch and epoch metrics. Run artefacts are stored under `example_logs/` (including TensorBoard event files and CSV exports), and console output is captured in the same directory when `output_catcher=True` is set on the writer.
If the example is executed inside a Git repository, a `git-info.json` file is created with the remote, branch, commit hash, and ready-to-run clone/checkout commands for reproducing the run.

Open TensorBoard to inspect the run:

```bash
tensorboard --logdir example_logs
```
