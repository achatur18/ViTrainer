from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Resize,
    CenterCrop,
)
from torchvision.datasets import ImageFolder
import yaml
import os

import torch
import torch.nn as nn

import torchvision

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", size=16)


def get_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train(config_path):
    config = get_config(config_path)
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(
        in_features=512, out_features=config["model"]["output_neurons"]
    )


train("/Users/abhaychaturvedi/Documents/Work/ViTrain/configs/classification.yaml")
