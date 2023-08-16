import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim
import cv2
from matplotlib import pyplot as plt
from image_show import image_show



def snr(hr_image, output_image):
    noise = hr_image - output_image
    return 10 * np.log10(np.mean(hr_image ** 2) / np.mean(noise ** 2))


def psnr(hr_image, output_image):
    mse = np.mean((hr_image - output_image) ** 2)
    return 10 * np.log10(1 / mse)


def ssim(hr_image, output_image, data_range, win_size=3):
    return compare_ssim(hr_image, output_image, data_range=data_range, win_size=win_size,channel_axis=-1)


def evaluate_test(model, loader, device):
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

            outputs = outputs.squeeze().cpu().numpy()
            hr = hr.squeeze().cpu().numpy()

            #test: show the image pairs
            lr = lr.squeeze().cpu().numpy()
            # for i in range(hr.shape[0]):
            #     plt.imshow(cv2.cvtColor(hr[i], cv2.COLOR_BGR2RGB))
            #     plt.show()
            #     plt.imshow(cv2.cvtColor(lr[i], cv2.COLOR_BGR2RGB))
            #     plt.show()

            for i in range(hr.shape[0]):
                snr_value = snr(hr[i], outputs[i])
                psnr_value = psnr(hr[i], outputs[i])
                ssim_value = ssim(hr[i], outputs[i], data_range=outputs[i].max() - outputs[i].min())

                snr_sum += snr_value
                psnr_sum += psnr_value
                ssim_sum += ssim_value

                num_samples += 1
            # print(i)
        image_show(hr[i],lr[i],outputs[i])
        plt.show()

    # Average
    snr_avg = snr_sum / num_samples
    psnr_avg = psnr_sum / num_samples
    ssim_avg = ssim_sum / num_samples
    #image_show()

    return snr_avg, psnr_avg, ssim_avg
