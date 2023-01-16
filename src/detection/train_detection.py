import argparse
import distutils.core
import json
import os
import random
import sys

import cv2
# Some basic setup:
# Setup detectron2 logger
import detectron2
import numpy as np
import torch
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

setup_logger()

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


def get_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Define the command-line flag
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="path to the input file")

# Parse the command-line arguments
args = parser.parse_args()
from detectron2.engine import HookBase

class EarlyStoppingHook(HookBase):
    def __init__(self, max_iter, threshold):
        self.max_iter = max_iter
        self.threshold = threshold
        self.best_loss = float("inf")
        self.no_improvement = 0

    def after_step(self):
        current_loss = self.trainer.iter_info.loss.cpu()
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.no_improvement = 0
        else:
            self.no_improvement += 1
            if self.no_improvement >= self.max_iter:
                print("Validation loss has not improved for {} iterations. Stopping training.".format(self.max_iter))
                self.trainer.stop = True



def train_detection(config_path):
    config = get_config(config_path)
    cfg = get_cfg()
    cfg.merge_from_file(
        os.path.join(os.getcwd(), "configs/mask_rcnn_R_50_C4_3x.yaml"))

    train_data_name = "{}_train".format(config["DATASET"]["name"])

    if train_data_name not in DatasetCatalog.list():
        register_coco_instances(
            train_data_name,
            {},
            config["DATASET"]["annotations_path"],
            config["DATASET"]["images_path"])
    dataset_dicts = DatasetCatalog.get(
        "{}_train".format(config["DATASET"]["name"]))
    metadata_ = MetadataCatalog.get(
        "{}_train".format(
            config["DATASET"]["name"]))

    # The "RoIHead batch size". 128 is faster, and good enough for this toy
    # dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata_.thing_classes)
    cfg.DATALOADER.NUM_WORKERS = config["DATALOADER"]["NUM_WORKERS"]
    cfg.SOLVER.IMS_PER_BATCH = config["SOLVER"]["IMS_PER_BATCH"]

    cfg.MODEL.MASK_ON = config["MODEL"]["MASK_ON"]
    cfg.DATASETS.TRAIN = (train_data_name,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.DEVICE = config["MODEL"]["DEVICE"]
    cfg.SOLVER.BASE_LR = config["BASE_LR"]
    cfg.SOLVER.MAX_ITER = config["EPOCH"]
    trainer = DefaultTrainer(cfg)

    trainer.register_hooks([EarlyStoppingHook(max_iter=cfg.SOLVER.MAX_ITER, threshold=0.01)])
    trainer.resume_or_load(resume=config["RESUME"])
    trainer.train()


train_detection(str(args.file))
