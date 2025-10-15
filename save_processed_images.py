import os
import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

from transforms.signal_to_image import (
    load_signal_from_txt, get_rp, get_mtf, get_stft_img, get_wvd_img, resize_img
)

def save_images_from_source(
    raw_data_root,
    save_root,
    method='rp',
    test_ratio=0.2
):
    method_map = {
        'rp': get_rp,
        'mtf': get_mtf,
        'stft': get_stft_img,
        'wvd': get_wvd_img
    }

    samples = [f for f in os.listdir(raw_data_root) if os.path.isdir(os.path.join(raw_data_root, f))]
    train_samples, test_samples = train_test_split(samples, test_size=test_ratio, random_state=42)

    for phase, phase_samples in [('train', train_samples), ('test', test_samples)]:
        for sample_name in phase_samples:
            sample_path = os.path.join(raw_data_root, sample_name)
            txt_files = sorted(glob.glob(os.path.join(sample_path, '*.txt')))

            images = []
            for txt in txt_files:
                signal = load_signal_from_txt(txt)
                img = method_map[method](signal)
                images.append(img)

            merged_img = np.mean(images, axis=0)
            merged_img = resize_img(merged_img)
            img_uint8 = (merged_img * 255).astype(np.uint8)

            label = 'PCOS' if sample_name[0] in ['A', 'P'] else 'nonPCOS'
            save_dir = os.path.join(save_root, phase, label)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{sample_name}_{method}.png")
            cv2.imwrite(save_path, img_uint8)
            print(f"[✓] Saved {save_path}")


if __name__ == '__main__':
    # 血浆数据
    save_images_from_source('data/data_plasma', 'data/images_plasma/', method='rp', test_ratio=0.2)
    # 卵泡液数据
    save_images_from_source('data/data_follicular', 'data/mages_follicular/', method='rp', test_ratio=0.2)
