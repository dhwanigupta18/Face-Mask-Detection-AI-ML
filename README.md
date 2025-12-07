# 😷 Face Mask Detection using AI/ML (Web + Webcam App)

This project is a complete **AI/ML-based Face Mask Detection system** built using **TensorFlow, OpenCV, and Flask**, with both:
- ✅ Real-time **Webcam Detection (Website)**
- ✅ Image Upload Prediction (ML Inference)
- ✅ Full end-to-end ML pipeline (Training → Testing → Deployment)

This project is ideal for **AI/ML internships, final year projects, and real-world deployment demo.**

---

## 🚀 Features

- 🔍 Detects **Mask / No Mask** using a trained CNN model
- 📸 Real-time **webcam detection via browser**
- 🌐 Flask-based **web application**
- 📊 Trained using **Kaggle Face Mask Dataset**
- ⚡ Works on **Apple Silicon (M1/M2)** with Conda
- 📁 Clean project structure with GitHub support

---

## 🧠 Tech Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Computer Vision:** OpenCV  
- **Web Framework:** Flask  
- **Frontend:** HTML, CSS, JavaScript  
- **Environment:** Conda (Miniforge)

---

## 📁 Project Structure

```text
Face-Mask-Detection-AI-ML/
│
├── models/
│   └── mask_detector.h5
│
├── src/
│   ├── train.py
│   ├── realtime_detector.py
│   └── dataset_prep.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── script.js
│
├── app.py
├── requirements.txt
├── req_no_tf.txt
└── README.md
⚙️ Environment Setup (Apple Silicon / macOS)
1️⃣ Install Miniforge (Conda)
bash Miniforge3-MacOSX-arm64.sh

2️⃣ Create Environment
conda create -n tf310 python=3.10 -y
conda activate tf310

3️⃣ Install TensorFlow
conda install -c conda-forge tensorflow -y

4️⃣ Install Project Packages
pip install -r req_no_tf.txt

📊 Dataset Preparation

Create this folder structure:

dataset/
├── with_mask/
└── without_mask/


You can download the dataset from Kaggle:
https://www.kaggle.com/datasets

Then verify dataset:

python src/dataset_prep.py --data_dir dataset

🏋️ Model Training
python src/train.py --data_dir dataset --epochs 10 --batch_size 32 --model_out models/mask_detector.h5

🧪 Real-time Webcam Detection (Local)
python src/realtime_detector.py --model models/mask_detector.h5


Press q to exit webcam.

🌐 Web Application (Flask + Webcam)
Run Web App:
python app.py

Open in Browser:
http://127.0.0.1:5000


Allow camera access → Click Start Detection

🌍 Deployment (Public Website)

This project can be deployed on:

✅ Render (Flask backend + webcam)

✅ HuggingFace Spaces (Image Upload Version)

Deployment files:

requirements.txt

app.py

Procfile (for Render)

📌 Use Cases

Smart surveillance systems

COVID safety compliance tools

Entry monitoring automation

AI-based face analysis systems

👩‍💻 Author

Dhwani Gupta
AI/ML Engineering Student
GitHub: https://github.com/dhwanigupta18

⭐ If you like this project, please give it a ⭐ on GitHub!
