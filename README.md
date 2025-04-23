# 🧠 3D Stroke Digit Classifier

Classifying handwritten digits from **3D motion gesture data**, projected into 2D grids and processed with a **Convolutional Neural Network**.

> 🔍 Built as part of the "Pattern Recognition and Machine Learning" course @ LUT University  
> 👨‍💻 Contributors: Omer Ahmed, Chamath Wijerathne, Mihály Frei

<p align="center">
  <img src="https://img.shields.io/badge/Made%20With-Python-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Model-CNN-green?style=for-the-badge">
  <!-- <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"> -->
</p>

---

## 🖼️ Project Overview

This project explores digit classification from **3D motion stroke data**, using a **Convolutional Neural Network**.
Using only **X and Y** dimensions, we:
- projected strokes to 2D,
- interpolated the strokes,
- converted them to **28x28 grayscale images**,
- and classified them with a **CNN (TensorFlow)**.

---

## 🧪 Dataset

- Retrieved from **Leap Motion** sensor
- 100 samples per digit (0–9), each: `35 x 3` stroke coordinates  

> ⚠️ Dataset not included due to licensing.  

---

## ⚙️ Preprocessing Pipeline

1. **Read stroke CSVs**, label digits
2. **Ignore Z**, project (X,Y)
3. **Normalize** to [0, 1]
4. **Interpolate** between points → mimic digit shape
5. **Rescale** to 28x28
6. **Fix orientation** (vertical flip correction)
7. **Reshape** to `(samples, 28, 28, 1)` for CNN input

---

## 🧠 Model Architecture

> Implemented in TensorFlow

- `Conv2D` layers + ReLU activations
- `MaxPooling` for downsampling
- `Dropout` to prevent overfitting
- `Dense` + `Softmax` for 10-digit output
- **Data Augmentation**: zoom, rotate, shift
- **Early stopping** after 4 non-improving epochs

---

## 📊 Results

- ✅ **Accuracy**: 98.5% on validation set  
- 📉 **Loss**: as low as 6%  
- 🧩 **Confusion Matrix** showed key issues:  
  - Digit **1 misclassified as 7**
  - Digit **4 vs 9** confusion (style-related)
 
---

## 💡 Future Work
- Real-time digit recognition using Leap Motion
- Improve poor-quality samples (e.g. digit 1)
- Try deeper CNNs or pretrained models

---

## 📁 Project Structure
📦 3d-digit-classifier/  
├── digit_classifier.ipynb       
├── digit_classifier.py         
├── inference_test.py            
├── requirements.txt           
└── digits_3d/                 
