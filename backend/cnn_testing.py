from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_addons.layers import InstanceNormalization  # 🛠️ ADD THIS


# Load SSIM loss function
import tensorflow as tf

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))



# Load model with InstanceNormalization included
cnn_model = load_model(
    r"C:\Users\HP\OneDrive\Desktop\demo\trained_model\color_restoration_better_unet.h5",
    custom_objects={
        'ssim_loss': ssim_loss,
        'InstanceNormalization': InstanceNormalization  # 🛠️ ADD THIS
    },
    compile=False
)
cnn_model.compile(loss=ssim_loss)


# Preprocessing
def pre_process(image, gamma=0.3):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# Testing function
def test_cnn_model(image_path, model, output_path="enhanced_output.jpg"):
    img = cv2.imread(image_path)
    img = pre_process(img, gamma=0.3)
    img_resized = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    enhanced_img = model.predict(img_resized)[0]
    enhanced_img = (enhanced_img * 255).astype(np.uint8)

    cv2.imwrite(output_path, enhanced_img)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title("MSRCR Image")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    plt.title("CNN Enhanced Image")
    plt.show()

    print(f"Enhanced image saved as {output_path}")

# Run test
test_cnn_model("image1_enhanced.jpg", cnn_model, "enhanced_output.jpg")
