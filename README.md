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
```

**2. Download the Pre-trained Weights:**
Because the PyTorch weight files exceed GitHub's standard file limits, they are hosted in the Releases section.
Download the best_0.001.pt weight file from the Releases tab of this repository.
Place the downloaded file inside the inference/ folder.

**3. Launch the GUI:**

```Bash
cd inference
python inference_app.py
```

**☁️ Cloud Training Environment**
Due to the computational intensity of the Darknet-53 backbone, local hardware was bypassed. The entire experimental pipeline, including the dataset augmentation (Mosaic, MixUp, CutMix) and hyperparameter sweeps, was executed utilizing dual NVIDIA Tesla T4 GPUs on Kaggle.

The full training pipeline and evaluation steps can be reproduced using the .ipynb files provided in the /notebooks directory.

*01_Model_Training_Pipeline.ipynb

*02_Metrics_and_Evaluation.ipynb

**👥 Project Team**
This system was engineered as part of the MCE 415 (Mechatronics Engineering) curriculum at the Federal University of Technology (FUT) Minna.

Project Manager & Lead Engineer: Eje Obed Honour
