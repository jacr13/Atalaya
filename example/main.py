"""
Simple example of Atalaya with code from :
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
"""

import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


from atalaya import Logger


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def parse_args(args):
    parser = argparse.ArgumentParser()
    # Device arguments
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (0 is no random-seed)."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
    )

    # Data arguments
    parser.add_argument(
        "--num-samples", type=int, default=10000, help="Number of samples."
    )
    parser.add_argument(
        "--ratio", type=float, default=0.7, help="Ratio of data for training."
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Number of samples per batch."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )

    # Logger and Grapher arguments (using atalaya)
    # Logger
    parser.add_argument(
        "--logger-folder",
        type=str,
        default="",
        help="Where to save the trained model, leave empty to not save anything.",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        default=False,
        help="Dont display information in terminal",
    )
    parser.add_argument(
        "--logger-name",
        type=str,
        default="exp",
        help='First part of the logger name (e.g. "exp1".',
    )
    parser.add_argument(
        "--load-params", type=str, default="", help="Where to load the params. "
    )
    parser.add_argument(
        "--load-folder", type=str, default="", help="Where to load the model. "
    )
    # Grapher
    parser.add_argument(
        "--grapher",
        type=str,
        default="",
        help="Name of the grapher. Leave empty for no grapher",
    )
    # if visdom
    parser.add_argument(
        "--visdom-url",
        type=str,
        default="http://localhost",
        help="visdom URL (default: http://localhost).",
    )
    parser.add_argument(
        "--visdom-port", type=int, default="8097", help="visdom port (default: 8097)"
    )
    parser.add_argument(
        "--visdom-username", type=str, default="", help="Username of visdom server."
    )
    parser.add_argument(
        "--visdom-password", type=str, default="", help="Password of visdom server."
    )

    return parser.parse_args(args)


def get_data_loaders(num_samples, ratio, batch_size):
    inputs = torch.randn(num_samples, 1, 32, 32)

    targets = torch.randint(0, 10, (num_samples,))
    targets = targets.view(-1, 1)

    targets_onehot = torch.FloatTensor(num_samples, 10)
    targets_onehot.zero_()
    targets_onehot.scatter_(1, targets, 1)

    ratio = int(num_samples * ratio)
    train_data = TensorDataset(inputs[:ratio], targets_onehot[:ratio])
    test_data = TensorDataset(inputs[ratio:], targets_onehot[ratio:])

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, test_data_loader


def run(mode, epoch, data_loader, model, criterion, optimizer, args, logger):
    history = {key: [] for key in ["loss", "acc"]}

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        _, out_ind = torch.max(outputs, 1)
        _, tar_ind = torch.max(targets, 1)

        acc = (out_ind == tar_ind).sum().item() / inputs.shape[0]

        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # keep the history of the values
        history["loss"].append(loss.item())
        history["acc"].append(acc)

        # or log directly at each iteraction
        logger.add_scalar("{}Scalar_loss".format(mode), loss.item())
        logger.add_scalar("{}Scalar_acc".format(mode), acc)

    return history


def train(epoch, data_loader, model, criterion, optimizer, args, logger):
    model.train()

    history = run(
        "train", epoch, data_loader, model, criterion, optimizer, args, logger
    )

    history = logger.register_plots(history, epoch, prefix="train")
    return history["loss"]


def test(epoch, data_loader, model, criterion, args, logger):
    model.eval()

    with torch.no_grad():
        history = run("test", epoch, data_loader, model, criterion, None, args, logger)

    history = logger.register_plots(history, epoch, prefix="test")
    return history["loss"]


def main(args):
    # get args
    args = parse_args(args)

    # init logger
    logger = Logger(
        name=args.logger_name,
        path=args.logger_folder,
        verbose=(not args.no_verbose),
        grapher=args.grapher,
        server=args.visdom_url,
        port=args.visdom_port,
    )

    if args.load_folder or args.load_params:
        args = logger.restore_parameters(
            args.load_params if args.load_params else args.load_folder
        )
    else:
        # add parameters to the logger
        logger.add_parameters(args)

    # if GPU available -> use device == cuda
    cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if cuda else "cpu")

    # if random seed use it, args.seed == 0 is non random seed
    if args.seed:
        torch.manual_seed(args.seed)
        if cuda:
            torch.cuda.manual_seed(args.seed)

    # data
    train_loader, test_loader = get_data_loaders(
        args.num_samples, args.ratio, args.batch_size
    )

    # the model
    model = Net()
    model.to(args.device)
    logger.add("model", model)

    # the loss
    criterion = nn.MSELoss()

    # the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    logger.add("optimizer", optimizer)

    # load model and optimizer from another experiment
    if args.load_folder:
        logger.restore(args.load_folder)

    # Train model
    stop_early = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            epoch, train_loader, model, criterion, optimizer, args, logger
        )

        _ = test(epoch, test_loader, model, criterion, args, logger)

        # store a checkpoint and save if train_loss < min(all previous train_loss)
        # replace with validation loss if you have one
        best_train_loss = logger.store(train_loss)

        # updating stop_early if not improving in validation set
        if best_train_loss:
            stop_early = 0
        else:
            stop_early += 1

        if stop_early > args.patience:
            logger.info(
                (
                    "Stopped training because it hasn't improve "
                    "performance in training set for "
                    "{} epochs"
                ).format(args.patience)
            )
            break

    logger.info("Optimization Finished!")

    # do some test here (not needed here because we test during training)

    # close logger
    logger.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
