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
from detectron2.utils.visualizer import ColorMode, Visualizer
import time
setup_logger()

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
    
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
parser.add_argument("--image_path", type=str, help="path to the image file")

# Parse the command-line arguments
args = parser.parse_args()


def predict(image_path, predictor, metadata_, MASK_ON=False):
    im = cv2.imread(image_path)
    # format is documented at
    # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    # print(outputs)
    seg_masks=None
    if MASK_ON:
        seg_masks = outputs["instances"].pred_masks

    scores = []
    for score in outputs["instances"].scores:
        scores.append(float(score))

    boxes = []
    for box in outputs["instances"].pred_boxes:
        boxes.append([int(x) for x in box])

    classes = []
    for class_ in outputs["instances"].pred_classes:
        classes.append(int(class_))

    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata_,
                   scale=0.5,
                   # remove the colors of unsegmented pixels. This option is
                   # only available for segmentation models
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = out.get_image()[:, :, ::-1]
    cv2.imwrite("output/result.jpg", out)
    return {"boxes": boxes, "classes": classes, "scores": scores, "seg_masks": seg_masks}


def predict_detection(config_path, image_path):
    config = get_config(config_path)
    cfg = get_cfg()
    cfg.merge_from_file(
        os.path.join(os.getcwd(), "configs/mask_rcnn_R_50_C4_3x.yaml"))

    metadata_ = MetadataCatalog.get(
        "{}_train".format(
            config["DATASET"]["name"]))
    # The "RoIHead batch size". 128 is faster, and good enough for this toy
    # dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config["DATASET"]["num_classes"]
    cfg.DATALOADER.NUM_WORKERS = config["DATALOADER"]["NUM_WORKERS"]
    cfg.SOLVER.IMS_PER_BATCH = config["SOLVER"]["IMS_PER_BATCH"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"]
    cfg.OUTPUT_DIR = config["OUTPUT"]["OUTPUT_DIR"]
    cfg.MODEL.MASK_ON = config["MODEL"]["MASK_ON"]
    cfg.MODEL.DEVICE = config["MODEL"]["DEVICE"]
    cfg.SOLVER.BASE_LR = config["BASE_LR"]
    cfg.SOLVER.MAX_ITER = config["EPOCH"]
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    image_pred = predict(image_path, predictor, metadata_, cfg.MODEL.MASK_ON)

    # Convert the dictionary to a JSON string
    json_data = json.dumps(image_pred)

    # Save the JSON string to a file
    with open('output/result.json', 'w') as outfile:
        outfile.write(json_data)
    return image_pred
    # cv2.imwrite("output/output_pred.jpg", image_pred)



def predict_detection_batch(config_path):
    config = get_config(config_path)
    cfg = get_cfg()
    cfg.merge_from_file(
        os.path.join(os.getcwd(), "configs/mask_rcnn_R_50_C4_3x.yaml"))

    metadata_ = MetadataCatalog.get(
        "{}_train".format(
            config["DATASET"]["name"]))
    # The "RoIHead batch size". 128 is faster, and good enough for this toy
    # dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config["DATASET"]["num_classes"]
    cfg.DATALOADER.NUM_WORKERS = config["DATALOADER"]["NUM_WORKERS"]
    cfg.SOLVER.IMS_PER_BATCH = config["SOLVER"]["IMS_PER_BATCH"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"]
    cfg.OUTPUT_DIR = config["OUTPUT"]["OUTPUT_DIR"]
    cfg.MODEL.MASK_ON = config["MODEL"]["MASK_ON"]
    cfg.MODEL.DEVICE = config["MODEL"]["DEVICE"]
    cfg.SOLVER.BASE_LR = config["BASE_LR"]
    cfg.SOLVER.MAX_ITER = config["EPOCH"]
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    train_data_name = "{}_train".format(config["DATASET"]["name"])

    if train_data_name not in DatasetCatalog.list():
        register_coco_instances(
            train_data_name,
            {},
            config["DATASET"]["annotations_path"],
            config["DATASET"]["images_path"])

    dataset_dicts = DatasetCatalog.get(train_data_name)

    image_preds=[]
    for d in dataset_dicts[:10]:
        image_path = d["file_name"]
        image_pred = predict(image_path, predictor, metadata_, cfg.MODEL.MASK_ON)
        image_preds.append(image_pred)

    total_time = [float(d["prediction-time"]) for d in image_preds]

    # Convert the dictionary to a JSON string
    image_preds.append({"total-time": sum(total_time)})
    json_data = json.dumps(image_preds)

    # Save the JSON string to a file
    with open('output/result.json', 'w') as outfile:
        outfile.write(json_data)
    return {"total-time": sum(total_time), "model-size": sys.getsizeof(torch.load(cfg.MODEL.WEIGHTS))}
    # cv2.imwrite("output/output_pred.jpg", image_pred)


print(isinstance(args.image_path, type(None)))
if not isinstance(args.image_path, type(None)):
    print(
        predict_detection(str(args.file),
            image_path=args.image_path))
else:
    print(
        predict_detection_batch(str(args.file)))
