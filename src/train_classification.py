import argparse
import os

import colorama
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip,
                                    RandomResizedCrop, Resize, ToTensor)

matplotlib.rc("font", size=16)


def get_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train_vgg16(
    model,
    train_loader,
    test_loader,
    device,
    data_size,
    config,
    num_epochs=3,
    learning_rate=0.1,
    decay_learning_rate=False
):
    # Some models behave differently in training and testing mode (Dropout, BatchNorm)
    # so it is good practice to specify which behavior you want.
    model.train()

    # We will use the Adam with Cross Entropy loss
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=config["model"]["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    if config["model"]["decay_learning_rate"]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.85)

    # We make multiple passes over the dataset
    for epoch in range(num_epochs):
        print("=" * 40, "Starting epoch %d" % (epoch + 1), "=" * 40)

        if decay_learning_rate:
            scheduler.step()

        total_epoch_loss = 0.0
        # Make one pass in batches
        for batch_number, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

            if batch_number % 5 == 0:
                print("Batch %d/%d" % (batch_number, len(train_loader)))

        train_acc = accuracy(model, train_loader, device)
        test_acc = accuracy(model, test_loader, device)

        print(
            colorama.Fore.GREEN
            + "\nEpoch %d/%d, Loss=%.4f, Train-Acc=%d%%, Valid-Acc=%d%%"
            % (
                epoch + 1,
                num_epochs,
                total_epoch_loss / data_size,
                100 * train_acc,
                100 * test_acc,
            ),
            colorama.Fore.RESET,
        )


def accuracy(model, data_loader, device):
    model.eval()

    num_correct = 0
    num_samples = 0
    with torch.no_grad():  # deactivates autograd, reduces memory usage and speeds up computations
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)

            # find the class number with the largest output
            predictions = torch.argmax(model(data), 1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(predictions)

    return num_correct / num_samples


def train(config_path):
    config = get_config(config_path)

    # Create datasets
    train_data = ImageFolder(
        config["dataset"]["train_path"],
        transform=Compose(
            [RandomResizedCrop(224), RandomHorizontalFlip(),
             ToTensor()]  # data augmentation
        ),
    )

    test_data = ImageFolder(
        config["dataset"]["test_path"],
        # give images the same size as the train images
        transform=Compose([Resize(256), CenterCrop(224), ToTensor()]),
    )

    # Our datasets have two classes:
    class_names = train_data.classes

    # Specify corresponding batched data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True)
    test_loader = DataLoader(
        test_data,
        batch_size=config["testing"]["batch_size"],
        shuffle=False)

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(
        in_features=512, out_features=len(class_names)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_vgg16(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=config["training"]["epochs"],
        data_size=len(train_loader),
        config=config)
    torch.save(
        model,
        os.path.join(
            config["output"]["directory"],
            config["output"]["weights_name"]))


# Define the command-line flag
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="path to the input file")

# Parse the command-line arguments
args = parser.parse_args()


train(config_path=str(args.file))
# train("/Users/abhaychaturvedi/Documents/Work/ViTrain/configs/classification.yaml")
