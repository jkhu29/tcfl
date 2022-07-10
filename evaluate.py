import os 
import argparse

import torch
import torchvision
from skimage.metrics import structural_similarity

import models
import utils
from dataset import load_image


def get_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--epoch', type=int, default=0, help='')
    opt = parser.parse_args()
    return opt


opt = get_options()

model = models.DnCNN(channels=1)
model.load_state_dict(torch.load("./ckpt/generator/epoch_{}.pth".format(opt.epoch)))
model.cuda()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.CenterCrop(256),
])

DUKE17_ROOT = "./data/Sparsity_SDOCT_DATASET_2012/"
# DUKE28_ROOT = "./data/Final_Publication_2013_SBSDI_sourcecode_withpcode/For synthetic experiments"

total_psnr = utils.AverageMeter()
total_ssim = utils.AverageMeter()
total_gcmse = utils.AverageMeter()

noise_psnr = utils.AverageMeter()
noise_ssim = utils.AverageMeter()
noise_gcmse = utils.AverageMeter()

for idx in os.listdir(DUKE17_ROOT):
    if idx != "Readme.txt":
        idx_root = os.path.join(DUKE17_ROOT, idx)

        input_path = os.path.join(idx_root, idx+"_Raw Image.tif")
        # input_path = os.path.join(idx_root, "test.tif")
        label_path = os.path.join(idx_root, idx+"_Averaged Image.tif")
        # label_path = os.path.join(idx_root, "average.tif")

        input_image = load_image(input_path)
        label_image = load_image(label_path)

        input_torch = transform(input_image).unsqueeze(0).cuda()
        label_torch = transform(label_image)

        with torch.no_grad():
            noise_torch = model(input_torch).clamp(-1, 1)
        image_torch = input_torch - noise_torch
        image_torch = image_torch.squeeze().cpu().clamp(0., 1.)

        noise_psnr.update(utils.calc_psnr(input_torch.squeeze().cpu(), label_torch.squeeze()))
        noise_ssim.update(structural_similarity(label_torch.numpy().squeeze(), input_torch.cpu().numpy().squeeze()))
        noise_gcmse.update(utils.calc_gcmse(label_torch.numpy().squeeze(), input_torch.cpu().numpy().squeeze())[0])

        total_psnr.update(utils.calc_psnr(image_torch.squeeze().cpu(), label_torch.squeeze()))
        total_ssim.update(structural_similarity(label_torch.numpy().squeeze(), image_torch.numpy().squeeze()))
        total_gcmse.update(utils.calc_gcmse(label_torch.numpy().squeeze(), image_torch.numpy().squeeze())[0])

print(
    " noise_psnr: "+str(noise_psnr.avg)+"\n",
    "noise_ssim: "+str(noise_ssim.avg)+"\n",
    "noise_gcmse: "+str(noise_gcmse.avg)+"\n",
    "total_psnr: "+str(total_psnr.avg)+"\n",
    "total_ssim: "+str(total_ssim.avg)+"\n",
    "total_gcmse: "+str(total_gcmse.avg)+"\n",
)