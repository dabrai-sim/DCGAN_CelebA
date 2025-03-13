# DCGAN_CelebA

# DCGAN: Deep Convolutional Generative Adversarial Network

This project implements a **Deep Convolutional GAN (DCGAN)** to generate realistic images using the **CelebA dataset**.

## Dataset
- Download the **CelebA dataset** from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- Extract the dataset and update `DATA_PATH` in `dcgan_training.py`.
- Preprocessing:
  - Resize images to **64x64**.
  - Normalize pixel values to **[-1,1]**.

## Model Architecture
### **Generator (G)**
- Takes random noise and generates images using **transposed convolutions**.

### **Discriminator (D)**
- Classifies real vs. generated images using **convolutional layers**.

## Training
Run the following command:
```bash
python dcgan_training.py
