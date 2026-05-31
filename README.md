# AI-Based Sign Language Translator 🤖

An AI-powered system designed to bridge communication gaps by recognizing and translating Sign Language gestures into text in real-time using Computer Vision and Deep Learning.

## 🚀 Project Overview
This project implements an end-to-end Machine Learning pipeline—from training a custom model to testing its metrics and running real-time predictions. It captures hand gestures via a camera feed, processes the visual data, and instantly outputs the corresponding alphanumeric text.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries & Frameworks:** OpenCV, MediaPipe, NumPy, TensorFlow/Keras

---

## 📂 Codebase Structure

The core functionality is split into three main scripts:

### 1. 🏋️‍♂️ `train.py`
* **Purpose:** Handles the model training pipeline.
* **Details:** Loads the processed gesture datasets, defines the neural network architecture, handles data augmentation, tunes hyperparameters, and saves the trained weights/model file upon completion.

### 2. 🧪 `test.py`
* **Purpose:** Evaluates the model's performance on unseen data.
* **Details:** Computes crucial metrics such as validation accuracy, loss curves, confusion matrices, and precision/recall to ensure the system is highly accurate before deployment.

### 3. 🔮 `predict.py`
* **Purpose:** The real-time inference and deployment script.
* **Details:** Integrates with OpenCV to access the webcam feed, captures live hand landmarks, feeds the processed frames into the trained model, and displays the translated text live on-screen.

---
