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


def train(config_path):
    config = get_config(config_path)

    # Create datasets
    train_data = ImageFolder(
        model["dataset"]["train_path"],
        transform=Compose(
            [RandomResizedCrop(224), RandomHorizontalFlip(),
             ToTensor()]  # data augmentation
        ),
    )

    test_data = ImageFolder(
        model["dataset"]["test_path"],
        # give images the same size as the train images
        transform=Compose([Resize(256), CenterCrop(224), ToTensor()]),
    )

    # Our datasets have two classes:
    class_names = train_data.classes

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(
        in_features=512, out_features=len(class_names)
    )


train("/Users/abhaychaturvedi/Documents/Work/ViTrain/configs/classification.yaml")
