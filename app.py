# =========================================
# IMPORTS
# =========================================

import io
import json
import cv2
import numpy as np

from flask import Flask, render_template, request, redirect
from PIL import Image
from onnx_infer import ONNXModel
from llm_ollama import generate_explanation


# =========================================
# APP SETUP
# =========================================

app = Flask(__name__)

CLASSES_PATH = "./model/classes.json"
ONNX_PATH = "./model/model.onnx"

onnx_model = ONNXModel(ONNX_PATH)


# =========================================
# LOAD CLASS NAMES
# =========================================

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)


# =========================================
# IMAGE TRANSFORM
# =========================================

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)

    return image


# =========================================
# PREDICTION FUNCTION
# =========================================

def get_prediction(image_bytes):
    input_np = transform_image(image_bytes)

    outputs = onnx_model.infer(input_np)

    exp = np.exp(outputs - np.max(outputs))
    probs = exp / exp.sum(axis=1, keepdims=True)

    pred = np.argmax(probs)
    confidence = np.max(probs) * 100

    return class_names[pred], confidence


# =========================================
# ROUTES
# =========================================

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file')

        if not file:
            return redirect(request.url)

        img_bytes = file.read()

        prediction, confidence = get_prediction(img_bytes)
        
        # Enhanced GenAI explanation with confidence
        explanation = generate_explanation(prediction, confidence)

        return render_template(
            'result.html',
            name=prediction,
            confidence=f"{confidence:.2f}",
            explanation=explanation
        )


    return render_template('index.html')


# =========================================
# RUN APP
# =========================================

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)