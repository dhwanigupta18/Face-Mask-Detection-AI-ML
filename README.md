# 😷 Face Mask Detection using AI/ML (Web + Webcam App)

This project is a complete **AI/ML-based Face Mask Detection system** built using TensorFlow, OpenCV, and Flask.

It supports:

* ✅ Real-time Webcam Detection (Browser-based)
* ✅ Image-based Prediction (ML Inference)
* ✅ End-to-End Pipeline (Training → Testing → Deployment)

This project is suitable for **AI/ML internships, academic submissions, and real-world demos**.

---

# 🚀 Features

* 🔍 Detects **Mask / No Mask** using a trained deep learning model
* 📸 Real-time webcam detection via browser
* 🌐 Flask-based web application
* 📊 Trained on Face Mask dataset (Kaggle)
* ⚡ Works on macOS (Apple Silicon) and other platforms
* 📁 Clean and modular project structure

---

# 🧠 Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV
* **Web Framework:** Flask
* **Frontend:** HTML, CSS, JavaScript
* **Environment:** Conda / Virtual Environment

---

# 📁 Project Structure

```
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
```

---

# ⚙️ Environment Setup

## Option 1: macOS (Apple Silicon / M1/M2)

### 1️⃣ Install Miniforge (Conda)

```bash
bash Miniforge3-MacOSX-arm64.sh
```

### 2️⃣ Create Environment

```bash
conda create -n tf310 python=3.10 -y
conda activate tf310
```

### 3️⃣ Install TensorFlow

```bash
conda install -c conda-forge tensorflow -y
```

### 4️⃣ Install Remaining Dependencies

```bash
pip install -r req_no_tf.txt
```

---

## Option 2: Standard Python (Windows/Linux)

```bash
pip install -r requirements.txt
```

---

# 📊 Dataset Preparation

Create dataset structure:

```
dataset/
├── with_mask/
└── without_mask/
```

Download dataset from Kaggle.

Verify dataset:

```bash
python src/dataset_prep.py --data_dir dataset
```

---

# 🏋️ Model Training

```bash
python src/train.py --data_dir dataset --epochs 10 --batch_size 32 --model_out models/mask_detector.h5
```

---

# 🧪 Real-Time Webcam Detection (CLI)

```bash
python src/realtime_detector.py --model models/mask_detector.h5
```

Press **q** to exit webcam.

---

# 🌐 Web Application (Flask + Webcam)

Run the app:

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

Allow camera access → click **Start Detection**

---

# 🖥️ Command Line Execution

This project can be fully executed from the terminal:

```bash
# Train model
python src/train.py --data_dir dataset --epochs 10 --batch_size 32 --model_out models/mask_detector.h5

# Run real-time detection
python src/realtime_detector.py --model models/mask_detector.h5

# Run web application
python app.py
```

---

# 🔄 Workflow

1. Dataset is prepared and labeled
2. Model is trained using CNN (TensorFlow/Keras)
3. Faces are detected using OpenCV
4. Mask classification is performed
5. Results are displayed via webcam or web interface

---

# 📈 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# 📊 Results

* Model achieves high accuracy (depends on dataset quality)
* Real-time detection via webcam
* Web-based detection via browser

### Sample Output

(Add screenshots here for better evaluation)

---

# 🧠 Model Choice

A CNN-based deep learning model is used for classification.
It provides a good balance between **accuracy and real-time performance**.

---

# ⭐ Unique Features

* Real-time detection inside browser (Flask + webcam)
* End-to-end ML pipeline
* Dual interface (CLI + Web App)

---

# 🌍 Deployment

This project can be deployed on:

* ✅ Render (Flask backend)
* ✅ HuggingFace Spaces (image inference version)

Required files:

* requirements.txt
* app.py
* Procfile (for Render)

---

# 📌 Use Cases

* Smart surveillance systems
* COVID safety compliance
* Entry monitoring systems
* AI-based face analysis

---

# 👩‍💻 Author

**Dhwani Gupta**
AI/ML Engineering Student

GitHub: https://github.com/dhwanigupta18

---

# ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!

