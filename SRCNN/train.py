import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os

from dataLoading import train_dataset, valid_dataset
from model import SRCNN
from evaluate import evaluate
from plot import Logger, plot

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print(device)

srcnn = SRCNN().to(device)

num_epochs = 100
batch_size = 16
lr = 0.0001

criterion = nn.MSELoss()
optimizer = Adam(srcnn.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model_save_dir = "saved_model"
os.makedirs(model_save_dir, exist_ok=True)
best_valid_psnr = 0

logger = Logger()

for epoch in range(num_epochs):
    srcnn.train()
    for hr, lr in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)

        # Forward
        outputs = srcnn(lr)
        loss = criterion(outputs, hr)

        # Back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    train_snr, train_psnr, train_ssim = evaluate(srcnn, train_loader, device)
    valid_snr, valid_psnr, valid_ssim = evaluate(srcnn, valid_loader, device)

    logger.append(loss.item(), train_snr, valid_snr, train_psnr, valid_psnr, train_ssim, valid_ssim)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: %.4f" % loss.item(),
          "Train SNR: %.4f" % train_snr.item(), "valid SNR: %.4f" % valid_snr,
          "Train PSNR: %.4f" % train_psnr, "Train SSIM: %.4f" % train_ssim, "Valid SNR: %.4f" % valid_snr,
          "Valid PSNR: %.4f" % valid_psnr, "Valid SSIM: %.4f" % valid_ssim)
    # print(
    #     f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Train SNR: {train_snr}, Train PSNR: {train_psnr}, Train SSIM: {train_ssim}, Valid SNR: {valid_snr}, Valid PSNR: {valid_psnr}, Valid SSIM: {valid_ssim}")

    if valid_psnr > best_valid_psnr:
        best_valid_psnr = valid_psnr
        model_save_path = os.path.join(model_save_dir, "best_model.pth")
        torch.save(srcnn.state_dict(), model_save_path)
        print(f"Model Update: New PSNR: {best_valid_psnr}")

    scheduler.step()

#logger.plot(logger)
logger.save_to_csv("training_data.csv")
plot(logger)