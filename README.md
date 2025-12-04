# Face Mask Detection (AI/ML) - Project
This repository contains a complete, ready-to-run Face Mask Detection project using TensorFlow/Keras and OpenCV.
It uses transfer learning (MobileNetV2) for classification and provides a real-time webcam detector.

## What is included
- `src/` : Python source code (training, model, utils, realtime detector).
- `notebooks/` : Project report and notes.
- `requirements.txt` : Python dependencies.
- `README.md` : This file.
- `dataset/` : **(Not included)** Put your dataset here as described below.
- `models/` : Trained model will be saved here after training.

## Dataset structure (place your images here)
```
dataset/
    with_mask/
        img1.jpg
        ...
    without_mask/
        img1.jpg
        ...
```

## Quick start
1. Create a virtual environment and install packages:
   ```
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

2. Prepare dataset in `dataset/` (see structure above).

3. Train (example):
   ```
   python src/train.py --data_dir dataset --epochs 10 --batch_size 32 --model_out models/mask_detector.h5
   ```

4. Run real-time detector (after training or use pre-trained model):
   ```
   python src/realtime_detector.py --model models/mask_detector.h5
   ```

## Notes
- Training on CPU can be slow. Use GPU if available.
- This repo uses MobileNetV2 transfer learning for better accuracy with smaller datasets.
