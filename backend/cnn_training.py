import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, ReLU, Add, Input, Activation, BatchNormalization, Dropout
import keras.backend as K
from tensorflow.image import ssim

# Define CNN Model with BatchNorm and Dropout
def build_cnn_model():
    inputs = Input(shape=(256, 256, 3))

    x = Conv2D(64, (3, 3), padding='same', activation=None)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(5):
        skip = x
        x = Conv2D(64, (3, 3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = Add()([x, skip])
        x = ReLU()(x)
        x = Dropout(0.3)(x)  # Dropout layer to reduce overfitting

    outputs = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)
    model = Model(inputs, outputs)

    # Loss function: MSE + SSIM
    def ssim_loss(y_true, y_pred):
        return 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=lambda y_true, y_pred: 0.8 * K.mean(K.square(y_true - y_pred)) + 0.2 * ssim_loss(y_true, y_pred)
    )

    return model

# Data Augmentation Function
def augment_image(image):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)  # Horizontal flip
    if np.random.rand() < 0.5:
        image = cv2.GaussianBlur(image, (3, 3), 0)  # Blur
    return image

# Load Dataset with Augmentation
def load_dataset(original_path, enhanced_path, img_size=(256, 256)):
    X_train, Y_train = [], []
    original_images = sorted(os.listdir(original_path))
    enhanced_images = sorted(os.listdir(enhanced_path))

    for orig_img, enh_img in zip(original_images, enhanced_images):
        orig = cv2.imread(os.path.join(original_path, orig_img))
        enh = cv2.imread(os.path.join(enhanced_path, enh_img))

        if orig is None or enh is None:
            continue

        orig = cv2.resize(orig, img_size).astype(np.float32) / 255.0
        enh = cv2.resize(enh, img_size).astype(np.float32) / 255.0

        orig = augment_image(orig)  # Apply augmentation
        enh = augment_image(enh)

        X_train.append(orig)
        Y_train.append(enh)

    return np.array(X_train), np.array(Y_train)

# Train Model
original_path = r"data/low"
enhanced_path = r"data/high"
X_train, Y_train = load_dataset(original_path, enhanced_path)

cnn_model = build_cnn_model()

# Reduce LR when validation loss stops improving
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

cnn_model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler])
cnn_model.save("model.h5")