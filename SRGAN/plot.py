import matplotlib.pyplot as plt
import csv


class Logger:
    def __init__(self):
        self.content_loss_values = []
        self.adversarial_loss_values = []
        self.train_snr_values = []
        self.valid_snr_values = []
        self.train_psnr_values = []
        self.valid_psnr_values = []
        self.train_ssim_values = []
        self.valid_ssim_values = []

    def append(self, content_loss, adversarial_loss, train_snr, valid_snr, train_psnr, valid_psnr, train_ssim,
               valid_ssim):
        self.content_loss_values.append(content_loss)
        self.adversarial_loss_values.append(adversarial_loss)
        self.train_snr_values.append(train_snr)
        self.valid_snr_values.append(valid_snr)
        self.train_psnr_values.append(train_psnr)
        self.valid_psnr_values.append(valid_psnr)
        self.train_ssim_values.append(train_ssim)
        self.valid_ssim_values.append(valid_ssim)

    def save_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Content Loss', 'Adversarial Loss', 'Train SNR', 'Valid SNR', 'Train PSNR',
                          'Valid PSNR', 'Train SSIM', 'Valid SSIM']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.content_loss_values)):
                writer.writerow({'Epoch': i + 1,
                                 'Content Loss': self.content_loss_values[i],
                                 'Adversarial Loss': self.adversarial_loss_values[i],
                                 'Train SNR': self.train_snr_values[i],
                                 'Valid SNR': self.valid_snr_values[i],
                                 'Train PSNR': self.train_psnr_values[i],
                                 'Valid PSNR': self.valid_psnr_values[i],
                                 'Train SSIM': self.train_ssim_values[i],
                                 'Valid SSIM': self.valid_ssim_values[i]})


def plot(logger):
    epochs = list(range(1, len(logger.content_loss_values) + 1))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, logger.content_loss_values, label="Content Loss")
    plt.plot(epochs, logger.adversarial_loss_values, label="Adversarial Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, logger.train_snr_values, label="Train SNR")
    plt.plot(epochs, logger.valid_snr_values, label="Valid SNR")
    plt.xlabel("Epoch")
    plt.ylabel("SNR")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, logger.train_psnr_values, label="Train PSNR")
    plt.plot(epochs, logger.valid_psnr_values, label="Valid PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, logger.train_ssim_values, label="Train SSIM")
    plt.plot(epochs, logger.valid_ssim_values, label="Valid SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()

    plt.tight_layout()
    plt.savefig("result_plot.png")
    plt.show()
