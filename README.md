# Face Mask Detection System

A real-time computer vision system that detects whether a person is wearing a face mask. This project uses Deep Learning (Transfer Learning with MobileNetV2) to classify faces in video streams with high accuracy and low latency.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## Project Overview

The goal of this project is to provide an efficient way to monitor public health safety compliance. It uses a two-stage detection pipeline:

1. Face Detection: Locates faces in the frame using a pre-trained SSD (Single Shot Multibox Detector) model.
2. Mask Classification: Cropped faces are passed to a fine-tuned MobileNetV2 model to predict "Mask" or "No Mask".

## Key Features

- Real-Time Detection: Works seamlessly with a standard webcam.
- High Accuracy: Uses Transfer Learning on the robust MobileNetV2 architecture.
- Dual Modes: Supports both live video streams and static image files.
- Visual Feedback: Displays color-coded bounding boxes (Green for Mask, Red for No Mask) with confidence scores.

## Tech Stack

- Language: Python 3.x
- Deep Learning: PyTorch, torchvision
- Computer Vision: OpenCV (cv2), imutils
- Data Processing: NumPy, PIL
- Model: MobileNetV2 (Pre-trained on ImageNet)

## Dataset

The model was trained on a dataset containing images of people with and without face masks.
Note: The dataset is not included in this repository due to size constraints.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/RaghavonGit/Face_Recognition.git](https://github.com/RaghavonGit/Face_Recognition.git)
   cd Face_Recognition
