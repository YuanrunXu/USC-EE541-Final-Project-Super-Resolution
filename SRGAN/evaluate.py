import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def snr(hr_image, output_image):
    noise = hr_image - output_image
    return 10 * np.log10(np.mean(hr_image ** 2) / np.mean(noise ** 2))


def psnr(hr_image, output_image):
    mse = np.mean((hr_image - output_image) ** 2)
    return 10 * np.log10(1 / mse)


def ssim(hr_image, output_image, data_range, win_size=3):
    return compare_ssim(hr_image, output_image, data_range=data_range, win_size=win_size, channel_axis=-1)


def evaluate(model, loader, device):
    model.eval()
    snr_sum = 0
    psnr_sum = 0
    ssim_sum = 0
    num_samples = 0

    with torch.no_grad():
        for hr, lr in loader:
            lr = lr.to(device)
            hr = hr.to(device)

            outputs = model(lr)

            outputs = outputs.squeeze().cpu().numpy().transpose(0, 2, 3, 1)
            hr = hr.squeeze().cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(hr.shape[0]):
                snr_value = snr(hr[i], outputs[i])
                psnr_value = compare_psnr(hr[i], outputs[i], data_range=hr[i].max() - hr[i].min())
                ssim_value = compare_ssim(hr[i], outputs[i], data_range=hr[i].max() - hr[i].min(),win_size=3,channel_axis=-1)

                snr_sum += snr_value
                psnr_sum += psnr_value
                ssim_sum += ssim_value

                num_samples += 1

    # Average
    snr_avg = snr_sum / num_samples
    psnr_avg = psnr_sum / num_samples
    ssim_avg = ssim_sum / num_samples

    return snr_avg, psnr_avg, ssim_avg
