import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def snr(hr_image, output_image):
    noise = hr_image - output_image
    return 10 * np.log10(np.mean(hr_image ** 2) / np.mean(noise ** 2))


def psnr(hr_image, output_image):
    mse = np.mean((hr_image - output_image) ** 2)
    return 10 * np.log10(1 / mse)


def ssim(hr_image, output_image, data_range):
    return compare_ssim(hr_image, output_image, data_range=data_range, channel_axis=-1)