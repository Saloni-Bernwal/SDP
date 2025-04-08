import cv2
import numpy as np
import pywt
import runpy
import os

def wavelet_transform(image):
    channels = cv2.split(image)
    enhanced_channels = []
    for channel in channels:
        coeffs2 = pywt.dwt2(channel, 'db4')
        cA, (cH, cV, cD) = coeffs2
        cH *= 1.5
        cV *= 1.5
        cD *= 1.5
        enhanced_channel = pywt.idwt2((cA, (cH, cV, cD)), 'db4')
        enhanced_channel = np.clip(enhanced_channel, 0, 255).astype(np.uint8)
        enhanced_channels.append(enhanced_channel)
    return cv2.merge(enhanced_channels)

def apply_clahe(image):
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

def adaptive_msr(image, scales=[15, 101, 301], eps=1e-3):
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    l = l.astype(np.float32) + eps
    retinex_outputs = []
    for scale in scales:
        if scale % 2 == 0:
            scale += 1
        blurred = cv2.GaussianBlur(l, (scale, scale), 0)
        retinex = np.log(l) - np.log(blurred + eps)
        retinex_outputs.append(retinex)
    msr = np.mean(retinex_outputs, axis=0)
    msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    final_lab = cv2.merge((msr, a, b))
    return cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)

def enhance_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        raise Exception("Could not read the image.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    wavelet = wavelet_transform(img_rgb)
    clahe = apply_clahe(wavelet)
    amsr = adaptive_msr(clahe)
    cv2.imwrite("msrcr_output.jpg", amsr)

    # Run CNN
    runpy.run_path(r"C:\Users\HP\OneDrive\Desktop\abc\backend\cnn_testing.py")
    final = cv2.imread("enhanced_output.jpg")
    final_bgr = cv2.cvtColor(amsr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_bgr)


enhance_image(r"C:\Users\HP\OneDrive\Desktop\demo - Copy\demo2.png",r"C:\Users\HP\OneDrive\Desktop\abc\backend\output.png")