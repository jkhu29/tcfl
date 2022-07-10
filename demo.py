import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from skimage import io
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

import models
import utils
from dataset import load_image


model = models.DnCNN(channels=1)
# model = models.GeneratorUNet()
model.load_state_dict(torch.load("./ckpt/generator/epoch_600.pth"))
model.cuda()

input_image = load_image("./data/Final_Publication_2013_SBSDI_sourcecode_withpcode/test.tif")
label_image = load_image("./data/Final_Publication_2013_SBSDI_sourcecode_withpcode/average.tif")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.CenterCrop(256),
])

input_torch = transform(input_image).unsqueeze(0).cuda()
label_torch = transform(label_image)

with torch.no_grad():
    noise_torch = model(input_torch).clamp(-1, 1)
image_torch = input_torch - noise_torch
image_torch = image_torch.squeeze().cpu().clamp(0., 1.)

print(utils.calc_psnr(input_torch.squeeze().cpu(), label_torch.squeeze()))
print(structural_similarity(label_torch.numpy().squeeze(), input_torch.cpu().numpy().squeeze()))
print(utils.calc_gcmse(label_torch.numpy().squeeze(), input_torch.cpu().numpy().squeeze())[0])

print(utils.calc_psnr(image_torch.squeeze().cpu(), label_torch.squeeze()))
print(structural_similarity(label_torch.numpy().squeeze(), image_torch.numpy().squeeze()))
print(utils.calc_gcmse(label_torch.numpy().squeeze(), image_torch.numpy().squeeze())[0])

plt.subplot(2, 2, 1)
plt.imshow(input_torch.squeeze().cpu().numpy(), cmap="gray")
plt.subplot(2, 2, 2)
plt.imshow(image_torch.squeeze().numpy(), cmap="gray")
plt.subplot(2, 2, 3)
plt.imshow(label_torch.squeeze().numpy() - input_torch.squeeze().cpu().numpy(), cmap="gray")
plt.subplot(2, 2, 4)
plt.imshow(noise_torch.squeeze().cpu().numpy(), cmap="gray")
plt.show()
