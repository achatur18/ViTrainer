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
import sys
# Define the preprocessing pipeline
transform = Compose([Resize(256), CenterCrop(224), ToTensor()])

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def quantize_model(model):
    # Convert the model to quantized version
    quantized_model = torch.quantization.convert(torch.quantization.prepare(model))
    torch.save(quantized_model,"output/quantised_model_final.pth")

    return quantized_model

def predict(config_path, image_path):
    config = get_config(config_path)
    model = torch.load(
        os.path.join(
            config["output"]["directory"],
            config["output"]["weights_name"]))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model=quantize_model(model)
    # Open the image
    image=cv2.imread(image_path)
    image = Image.fromarray(image).convert('RGB')

    batch = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(batch)

    # Get the class with the highest probability
    _, pred = torch.max(output, 1)
    return {"pred":float(pred[0]), "model-size": model_size(model)}


def predict_batch(config_path, image_path):
    config = get_config(config_path)
    model = torch.load(
        os.path.join(
            config["output"]["directory"],
            config["output"]["weights_name"]))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model=quantize_model(model)
    # Open the image
    image=cv2.imread(image_path)
    image = Image.fromarray(image).convert('RGB')

    batch = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(batch)

    # Get the class with the highest probability
    _, pred = torch.max(output, 1)
    return {"pred":float(pred[0]), "model-size": model_size(model)}


# image_path = "/Users/abhaychaturvedi/Documents/Work/accelerators/classification_data/passport/14_1.png"

if __name__ == '__main__':

    # Define the command-line flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="path to the config file")
    parser.add_argument("--image_path", type=str, help="path to the image file")

    # Parse the command-line arguments
    args = parser.parse_args()

    print(isinstance(args.image_path, type(None)))
    if not isinstance(args.image_path, type(None)):
        print(
            predict(str(args.file),
                image_path=args.image_path))
    else:
        print(
            predict_batch(str(args.file)))
