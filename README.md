# AI-Powered COVID-19 X-ray Detection System with TensorRT & LLM Integration

---

## Overview

This project is a **containerized AI-powered medical imaging system** designed to detect COVID-19 from chest X-ray images and generate intelligent explanations using Large Language Models (LLMs).

The system leverages **ONNX Runtime with NVIDIA TensorRT acceleration** for high-performance inference on GPU, and integrates **Ollama (LLM)** to provide human-readable explanations, medical insights, and recommendations.

The complete pipeline is deployed using **Docker and Docker Compose**, enabling scalable and reproducible deployment.

---

## Features

* COVID-19 detection from chest X-ray images
* High-speed inference using **TensorRT optimization**
* ONNX-based model deployment
* LLM integration using Ollama (phi model)
* AI-generated explanation of predictions
* Flask-based web interface
* Fully containerized multi-service architecture
* Docker Compose orchestration

---

## Dataset

* Chest X-ray dataset (COVID-19 / Normal / Pneumonia)
* Preprocessed and normalized image data
* Used for training and inference

---

## Model Information

### Classification Model

| Component         | Value                      |
| ----------------- | -------------------------- |
| Model Type        | CNN-based classifier       |
| Framework         | PyTorch → ONNX             |
| Input Size        | 224 × 224                  |
| Output Classes    | COVID / Normal / Pneumonia |
| Deployment Format | ONNX                       |

---

## System Architecture

The system consists of the following components:

* Flask Backend (API + Web UI)
* ONNX Runtime + TensorRT Execution Provider
* Deep Learning Model (ONNX)
* NVIDIA GPU (CUDA acceleration)
* Ollama LLM Service
* Docker Containers
* Docker Compose Orchestration

---

## Inference Workflow

1. User uploads a chest X-ray image
2. Flask backend preprocesses the image
3. Image is passed to ONNX Runtime
4. TensorRT accelerates inference on GPU
5. Model returns predicted class
6. Result is sent to Ollama LLM
7. LLM generates:

   * Explanation of result
   * Possible causes
   * Medical insights
8. Final output is displayed in browser

---

## Project Structure

```id="trt1"
FINAL_PROJECT/
│
├── app.py
├── predict.py
├── onnx_infer.py
├── llm_ollama.py
├── requirements.txt
├── Dockerfile
├── docker/
│   └── docker-compose.yml
│
├── templates/
├── static/
│
└── model/   (download separately)
```

---

## Technologies Used

### Machine Learning

* PyTorch
* ONNX
* ONNX Runtime
* **TensorRT (GPU optimization)**

### Backend

* Flask
* NumPy
* OpenCV / PIL

### Generative AI

* Ollama
* Phi model

### Deployment

* Docker
* Docker Compose
* NVIDIA Container Toolkit

---

## Docker Deployment

### Prerequisites

* Docker Desktop
* NVIDIA GPU
* CUDA-compatible drivers
* NVIDIA Container Toolkit

---

## ▶Run the Application

```bash id="trt2"
docker-compose up -d
```

---

## Access Application

```id="trt3"
http://localhost:5000
```

---

## 🔌 Services

| Service    | Port  |
| ---------- | ----- |
| Flask App  | 5000  |
| Ollama API | 11434 |

---

## LLM Setup (Ollama)

```bash id="trt4"
docker exec -it ollama ollama pull phi
```

---

## Model Setup

Download ONNX model and place inside:

```id="trt5"
model/
```

(Provide dataset/model link here)

---

## GPU Acceleration (TensorRT)

* ONNX model is executed using **TensorRT Execution Provider**
* Enables:

  * Faster inference
  * Reduced latency
  * Optimized GPU utilization

---

## Example Output

The system provides:

* Predicted disease class
* Confidence score
* AI-generated explanation
* Medical guidance
---

## Future Improvements
* Multi-model deployment
* Cloud GPU deployment
* Real-time X-ray streaming
* Advanced medical explanation tuning

---

This project demonstrates a complete **end-to-end AI deployment pipeline** integrating:

* Deep Learning for medical diagnosis
* TensorRT-based GPU acceleration
* ONNX model optimization
* LLM-based explanation generation
* Containerized multi-service deployment

The system provides scalable, efficient, and explainable AI for medical imaging applications.

---

## License

MIT License
