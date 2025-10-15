# transforms/signal_to_image.py

import numpy as np
import cv2
from pyts.image import RecurrencePlot, MarkovTransitionField
from scipy.signal import stft
from tftb.processing import WignerVilleDistribution

# 读取.txt中第二列的拉曼强度
def load_signal_from_txt(txt_path):
    data = np.loadtxt(txt_path)
    return data[:, 1]  # shape: (1024,)

# Resize图像到224x224
def resize_img(img, size=(224, 224)):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

# RP
def get_rp(signal):
    rp = RecurrencePlot()
    img = rp.fit_transform(signal.reshape(1, -1))[0]
    return resize_img(img)

# MTF
def get_mtf(signal):
    mtf = MarkovTransitionField()
    img = mtf.fit_transform(signal.reshape(1, -1))[0]
    return resize_img(img)

# STFT
def get_stft_img(signal):
    f, t, Zxx = stft(signal)
    magnitude = np.abs(Zxx)
    return resize_img(magnitude)

# WVD
def get_wvd_img(signal):
    wvd = WignerVilleDistribution(signal)
    tfr, _, _ = wvd.run()
    img = np.abs(tfr)
    return resize_img(img)

