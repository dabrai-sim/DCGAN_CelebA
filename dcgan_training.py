"""
DCGAN Implementation for Generating Realistic Images

This script defines and trains a Deep Convolutional Generative Adversarial Network (DCGAN) 
using PyTorch. It includes dataset preprocessing, Generator and Discriminator models, 
training loop, and visualization of generated images.

Author: Simrann Dabrai
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset path
DATA_PATH = "datasets/extracted/img_align_celeba"

# Image transformations: Resize, Crop, Convert to Tensor, Normalize
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = ImageFolder(root=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)


# Generator Model
class Generator(nn.Module):
    """Generates images from random noise using transposed convolutions."""
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# Discriminator Model
class Discriminator(nn.Module):
    """Classifies images as real or fake using convolutional layers."""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
NUM_EPOCHS = 50
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

for epoch in range(NUM_EPOCHS):
    for i, (images, _) in enumerate(dataloader):
        # Train Discriminator
        real_images = images.to(device)
        real_labels = torch.ones(real_images.size(0), 1, device=device)
        fake_labels = torch.zeros(real_images.size(0), 1, device=device)

        optimizer_d.zero_grad()
        real_outputs = discriminator(real_images)
        loss_real = criterion(real_outputs, real_labels)

        noise = torch.randn(real_images.size(0), 100, 1, 1, device=device)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images.detach())
        loss_fake = criterion(fake_outputs, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        fake_outputs = discriminator(fake_images)
        loss_g = criterion(fake_outputs, real_labels)
        loss_g.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {i}/{len(dataloader)} "
                  f"Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    vutils.save_image(fake_images, os.path.join("output", f"epoch_{epoch+1}.png"), normalize=True)

print("Training completed.")
