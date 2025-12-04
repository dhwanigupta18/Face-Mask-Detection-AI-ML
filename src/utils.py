"""
utils.py
Utility functions for dataset loading and simple helpers.
"""
import os
import cv2
import numpy as np

def load_image(path, size=(224,224)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return img
