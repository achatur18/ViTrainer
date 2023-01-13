import argparse
import os

import cv2
import colorama
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip,
                                    RandomResizedCrop, Resize, ToTensor)

# Define the preprocessing pipeline
transform = Compose([Resize(256), CenterCrop(224), ToTensor()])


def get_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def predict(config_path, image_path):
    config = get_config(config_path)
    model = torch.load(
        os.path.join(
            config["output"]["directory"],
            config["output"]["weights_name"]))
    model.eval()
    # Open the image
    image=cv2.imread(image_path)
    image = Image.fromarray(image).convert('RGB')

    batch = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(batch)

    # Get the class with the highest probability
    _, pred = torch.max(output, 1)
    return float(pred[0])


# image_path = "/Users/abhaychaturvedi/Documents/Work/accelerators/classification_data/passport/14_1.png"


# Define the command-line flag
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="path to the config file")
parser.add_argument("--image_path", type=str, help="path to the image file")

# Parse the command-line arguments
args = parser.parse_args()

print(
    predict(
        args.file,
        args.image_path))
