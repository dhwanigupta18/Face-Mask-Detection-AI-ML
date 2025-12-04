# Project Report — Face Mask Detection

## Summary
Built a CNN-based face mask detector using transfer learning (MobileNetV2). The model classifies faces into `Mask` / `No Mask`. Real-time inference is implemented using OpenCV webcam + Haar Cascade face detection.

## Architecture
- Input: 224x224 RGB images
- Feature extractor: MobileNetV2 (pretrained on ImageNet, frozen)
- Classifier head: AveragePooling -> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(1, sigmoid)

## Training details (suggested)
- Loss: Binary crossentropy
- Optimizer: Adam lr=1e-4
- Epochs: 10-30 depending on dataset size
- Batch size: 16-64

## Improvements & next steps
- Unfreeze top MobileNet layers and fine-tune (with low LR)
- Use face detection (MTCNN or DNN) for better bounding boxes
- Add class weighting / focal loss for imbalanced datasets
- Export the model to TensorFlow SavedModel or TFLite for mobile deployment
