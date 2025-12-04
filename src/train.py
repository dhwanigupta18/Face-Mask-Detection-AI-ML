"""
train.py
Train a face mask detector using transfer learning (MobileNetV2).
Usage:
    python src/train.py --data_dir dataset --epochs 10 --batch_size 32 --model_out models/mask_detector.h5
"""
import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

def build_model(input_shape=(224,224,3)):
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7,7))(headModel)
    headModel = Flatten()(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    # freeze base model
    for layer in baseModel.layers:
        layer.trainable = False

    return model

def main(args):
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise ValueError("data_dir does not exist: %s" % data_dir)

    # ImageDataGenerator to load images from folders
    aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    trainGen = aug.flow_from_directory(
        data_dir,
        target_size=(224,224),
        batch_size=args.batch_size,
        class_mode="binary",
        subset="training"
    )

    valGen = aug.flow_from_directory(
        data_dir,
        target_size=(224,224),
        batch_size=args.batch_size,
        class_mode="binary",
        subset="validation"
    )

    model = build_model()
    opt = Adam(learning_rate=1e-4)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    checkpoint = ModelCheckpoint(args.model_out, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit(
        trainGen,
        steps_per_epoch=len(trainGen),
        validation_data=valGen,
        validation_steps=len(valGen),
        epochs=args.epochs,
        callbacks=[checkpoint]
    )

    print("Training complete. Best model saved to:", args.model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_out", default="models/mask_detector.h5", help="Output model path")
    args = parser.parse_args()
    main(args)
