import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os

from dataLoading import train_dataset, valid_dataset
from model import Generator, Discriminator
from evaluate import evaluate
from plot import Logger, plot

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print(device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

num_epochs = 100
batch_size = 16
lr = 0.0001

criterion_content = nn.MSELoss()
criterion_adversarial = nn.BCELoss()

optimizer_G = Adam(generator.parameters(), lr=lr)
optimizer_D = Adam(discriminator.parameters(), lr=lr)

scheduler_G = StepLR(optimizer_G, step_size=30, gamma=0.1)
scheduler_D = StepLR(optimizer_D, step_size=30, gamma=0.1)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model_save_dir = "saved_model"
os.makedirs(model_save_dir, exist_ok=True)
best_valid_psnr = 0

logger = Logger()

for epoch in range(num_epochs):
    for hr, lr in train_loader:
        hr = hr.to(device)
        lr = lr.to(device)
        batch_size = hr.size(0)

        # Train Discriminator
        optimizer_D.zero_grad()

        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        # Discriminator loss for real images
        real_output = discriminator(hr)
        real_loss = criterion_adversarial(real_output.squeeze(), real_labels)

        # Discriminator loss for fake images
        fake_images = generator(lr)
        fake_output = discriminator(fake_images.detach())
        fake_loss = criterion_adversarial(fake_output.squeeze(), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        # Generator content loss
        content_loss = criterion_content(fake_images, hr)

        # Generator adversarial loss
        fake_output = discriminator(fake_images)
        adversarial_loss = criterion_adversarial(fake_output.squeeze(), real_labels)

        g_loss = content_loss + 0.001 * adversarial_loss
        g_loss.backward()
        optimizer_G.step()

    # Evaluate
    train_snr, train_psnr, train_ssim = evaluate(generator, train_loader, device)
    valid_snr, valid_psnr, valid_ssim = evaluate(generator, valid_loader, device)

    logger.append(content_loss.item(), adversarial_loss.item(), train_snr, valid_snr, train_psnr, valid_psnr, train_ssim, valid_ssim)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Content Loss: {content_loss.item():.4f}, Adversarial Loss: {adversarial_loss.item():.4f}, Train SNR: {train_snr:.4f}, Train PSNR: {train_psnr:.4f}, Train SSIM: {train_ssim:.4f}, Valid SNR: {valid_snr:.4f}, Valid PSNR: {valid_psnr:.4f}, Valid SSIM: {valid_ssim:.4f}")

    if valid_psnr > best_valid_psnr:
        best_valid_psnr = valid_psnr
        model_save_path = os.path.join(model_save_dir, "best_generator.pth")
        torch.save(generator.state_dict(), model_save_path)
        print(f"Model Update: New Best PSNR: {best_valid_psnr}")

    scheduler_G.step()
    scheduler_D.step()

logger.save_to_csv("training_data.csv")
plot(logger)
