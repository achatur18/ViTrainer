

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
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import detectron2.data.transforms as T
import time

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def quantize_model(model):
    # Convert the model to quantized version
    quantized_model = torch.quantization.convert(torch.quantization.prepare(model))
    torch.save(quantized_model,"output/quantised_model_final.pth")

    return quantized_model
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
parser.add_argument("--image_path", type=str, help="path to the image file")

# Parse the command-line arguments
args = parser.parse_args()


class DefaultPredictor_:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model = quantize_model(self.model)
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def predict(image_path, predictor, metadata_):
    im = cv2.imread(image_path)
    # format is documented at
    # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    start_time=time.time()
    outputs = predictor(im)
    # print(outputs)

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
    return {"image_path":image_path, "boxes": boxes, "classes": classes, "scores": scores, "prediction-time":time.time()-start_time}


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
    predictor = DefaultPredictor_(cfg)

    image_pred = predict(image_path, predictor, metadata_)

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
    predictor = DefaultPredictor_(cfg)

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
        image_pred = predict(image_path, predictor, metadata_)
        image_preds.append(image_pred)

    total_time = [float(d["prediction-time"]) for d in image_preds]

    # Convert the dictionary to a JSON string
    image_preds.append({"total-time": sum(total_time)})
    json_data = json.dumps(image_preds)

    # Save the JSON string to a file
    with open('output/result.json', 'w') as outfile:
        outfile.write(json_data)
    return {"total-time": sum(total_time), "model-size": sys.getsizeof(torch.load("output/quantised_model_final.pth"))}
    # cv2.imwrite("output/output_pred.jpg", image_pred)

print(isinstance(args.image_path, type(None)))
if not isinstance(args.image_path, type(None)):
    print(
        predict_detection(str(args.file),
            image_path=args.image_path))
else:
    print(
        predict_detection_batch(str(args.file)))
