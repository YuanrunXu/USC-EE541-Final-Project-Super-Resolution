import torch
from torch.utils.data import DataLoader

from model import SRCNN
from dataLoading import test_dataset
from evaluate_test import evaluate_test

device = torch.device("mps")
print(device)

srcnn = SRCNN().to(device)
srcnn.load_state_dict(torch.load("./saved_model/best_model.pth"))

batch_size = 16
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

snr_avg, psnr_avg, ssim_avg = evaluate_test(srcnn, test_loader, device)
print(f"Test: SNR: {snr_avg}, PSNR: {psnr_avg}, SSIM: {ssim_avg}")
