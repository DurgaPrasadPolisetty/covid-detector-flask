# =========================================
# IMPORTS
# =========================================

import json
import numpy as np
import cv2
from PIL import Image

from onnx_infer import ONNXModel


# =========================================
# CONFIG
# =========================================

ONNX_PATH = "./model/model.onnx"
CLASSES_PATH = "./model/classes.json"


# =========================================
# LOAD MODEL
# =========================================

onnx_model = ONNXModel(ONNX_PATH)

print("ONNX model loaded successfully")


# =========================================
# LOAD CLASS NAMES
# =========================================

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)


# =========================================
# IMAGE TRANSFORM (NO TORCH)
# =========================================

def transform_image(image):
    image = Image.fromarray(image).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)

    return image


# =========================================
# PREDICTION FUNCTION
# =========================================

def predict(image):

    # Handle channels
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Transform
    input_tensor = transform_image(image)

    # ONNX inference
    outputs = onnx_model.infer(input_tensor)

    # Softmax
    exp = np.exp(outputs - np.max(outputs))
    probs = exp / exp.sum(axis=1, keepdims=True)

    pred = np.argmax(probs)
    confidence = np.max(probs) * 100

    pred_class = class_names[pred]

    print(f"Prediction: {pred_class}, Confidence: {confidence:.2f}%")

    return pred_class, confidence