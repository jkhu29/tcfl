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


model = models.DnCNN(channels=1)
model.load_state_dict(torch.load("./ckpt/generator/epoch_100.pth"))
model.cuda()

# input_numpy = io.imread("./data/Final_Publication_2013_SBSDI_sourcecode_withpcode/test.tif")
# label_numpy = io.imread("./data/Final_Publication_2013_SBSDI_sourcecode_withpcode/average.tif")

input_numpy = io.imread("./data/Final_Publication_2013_SBSDI_sourcecode_withpcode/For synthetic experiments/2/test.tif")
label_numpy = io.imread("./data/Final_Publication_2013_SBSDI_sourcecode_withpcode/For synthetic experiments/2/average.tif")

input_image = Image.fromarray(input_numpy)
label_image = Image.fromarray(label_numpy)
input_torch = torchvision.transforms.ToTensor()(input_image).unsqueeze(0).to("cuda")
label_torch = torchvision.transforms.ToTensor()(label_image).squeeze()
with torch.no_grad():
    noise_torch = model(input_torch)
image_torch = input_torch - noise_torch
image_torch = image_torch.clamp(0, 1).squeeze().cpu()

print(cv2.PSNR(label_torch.numpy().squeeze(), input_torch.cpu().numpy().squeeze(), R=1.))
print(structural_similarity(label_torch.numpy().squeeze(), input_torch.cpu().numpy().squeeze()))
print(utils.calc_gcmse(label_torch.numpy().squeeze(), input_torch.cpu().numpy().squeeze())[0])

print()

print(cv2.PSNR(label_torch.numpy().squeeze(), image_torch.numpy().squeeze(), R=1.))
print(structural_similarity(label_torch.numpy().squeeze(), image_torch.numpy().squeeze()))
print(utils.calc_gcmse(label_torch.numpy().squeeze(), image_torch.numpy().squeeze())[0])

plt.subplot(2, 2, 1)
plt.imshow(input_torch.squeeze().cpu().numpy(), cmap="gray")
plt.subplot(2, 2, 2)
plt.imshow(image_torch.squeeze().numpy(), cmap="gray")
plt.subplot(2, 2, 3)
plt.imshow(label_torch.squeeze().numpy(), cmap="gray")
plt.subplot(2, 2, 4)
plt.imshow(noise_torch.squeeze().cpu().numpy(), cmap="gray")
plt.show()
