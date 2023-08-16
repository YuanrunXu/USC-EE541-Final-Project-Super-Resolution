import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from dataLoading import test_dataset
import matplotlib.pyplot as plt

from evaluate import snr, psnr, ssim


def bicubic(loader):
    snr_sum = 0
    psnr_sum = 0
    ssim_sum = 0
    num_samples = 0

    for hr, lr in loader:
        hr = hr.squeeze().numpy().transpose(1, 2, 0)
        lr = lr.squeeze().numpy().transpose(1, 2, 0)
        lr = np.array(lr)

        output = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

        snr_value = snr(hr, output)
        psnr_value = psnr(hr, output)
        ssim_value = ssim(hr, output, data_range=output.max() - output.min())

        snr_sum += snr_value
        psnr_sum += psnr_value
        ssim_sum += ssim_value
        num_samples += 1

    # Show
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(hr)
    axs[0].set_title("HR Image")
    axs[1].imshow(lr)
    axs[1].set_title("LR Image")
    axs[2].imshow(output)
    axs[2].set_title("Bicubic")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

    snr_avg = snr_sum / num_samples
    psnr_avg = psnr_sum / num_samples
    ssim_avg = ssim_sum / num_samples

    return snr_avg, psnr_avg, ssim_avg


if __name__ == "__main__":
    snr_avg, psnr_avg, ssim_avg = bicubic(test_dataset)
    print(f"Bicubic: SNR: {snr_avg:.2f}, PSNR: {psnr_avg:.2f}, SSIM: {ssim_avg:.4f}")
