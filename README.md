# YOLOv3 Autonomous Road Anomaly Detection 🚗🛣️

![Qualitative Results](assets/Figure3_QualitativeGrid.png)

## Overview
This repository contains an end-to-end computer vision pipeline utilizing the **YOLOv3 (Darknet-53)** architecture to detect and localize road surface degradation. The model was trained to identify three distinct hazard classes: Potholes, Speedbumps, and Cracks. 

A custom Tkinter Graphical User Interface (GUI) is included to run real-time inference on unseen images.

## 📊 Model Performance Metrics
The network was optimized via an extensive learning rate sweep (0.1 to 0.00001). The optimal convergence was achieved at **LR = 0.001**, yielding the following metrics:

* **Global mAP@0.5:** 61.55%
* **Global Precision:** 78.91%
* **Speedbump mAP:** 93.84%
* **Pothole mAP:** 73.88%
* **Crack mAP:** 14.82%

*(See `assets/Figure4_PerClass.png` for the full visual breakdown).*

## 🧠 Architecture
This project utilizes the YOLOv3 object detection framework featuring:
1. **Darknet-53 Backbone:** A 53-layer residual network for spatial feature extraction.
2. **Feature Pyramid Network (FPN) Neck:** For multi-scale anomaly detection (e.g., massive speedbumps vs. hairline cracks).
3. **Computational Footprint:** 268 Layers | 68.2M Parameters | 155.8 GFLOPs.

## 📂 Dataset
The model was trained on an aggregated Unified Road Dataset. The data was programmatically partitioned into a strict 70/15/15 (Train/Val/Test) split. 
* 🔗 [Link to Kaggle Dataset](#) *(Insert your Kaggle link here)*
* **Note:** The dataset is not hosted in this repository due to size constraints.

## 🚀 How to Run the Inference App

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/YourUsername/YOLOv3-Road-Anomaly-Detection.git](https://github.com/YourUsername/YOLOv3-Road-Anomaly-Detection.git)
cd YOLOv3-Road-Anomaly-Detection
pip install -r requirements.txt
