from io import BytesIO

import cv2
import numpy as np
import yaml
from fastapi import (Body, Depends, FastAPI, File, HTTPException, Request,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.classification.predict_classification import \
    predict as predicting_classification
from src.classification.train_classification import \
    train_classification as training_classification
from src.detection.predict_detection import \
    predict_detection as predicting_detection
from src.detection.train_detection import train_detection as training_detection

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/train/detection")
async def train_detection(config_file: bytes = File()):
    config_file = yaml.safe_load(config_file)
    training_detection(config_file)


@app.post("/train/classification")
async def train_classification(config_file: bytes = File()):
    config_file = yaml.safe_load(config_file)
    training_classification(config_file)


@app.post("/predict/detection")
async def predict_detection(config_file: bytes = File(), image_data: UploadFile = File(...)):
    image_data = await image_data.read()
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    config_file = yaml.safe_load(config_file)
    return predicting_detection(config_file, image)


@app.post("/predict/classification")
async def predict_classification(config_file: bytes = File(), image_data: UploadFile = File(...)):
    image_data = await image_data.read()
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    config_file = yaml.safe_load(config_file)
    pred = predicting_classification(config_file, image)
    return {"class": pred}
