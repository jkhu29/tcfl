import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def weights_init(model):
    """init from article"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1, std=0.02)
            nn.init.constant_(m.bias, 0)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calc_ssim(img1, img2, window_size=11):
    """calculate SSIM"""
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average=True)


def calc_psnr(img1, img2, R: float = 1.0):
    """calculate PNSR on cuda and cpu"""
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(R / torch.sqrt(mse))


def calc_mse(img1, img2):
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return mse.item()


def calc_epi(img1, img2):
    img1U = img1[1:-1, :]
    img1D = img1[2:, :]
    img1L = img1[:, 1:-1] 
    img1R = img1[:, 2:]

    img2U = img2[1:-1, :]
    img2D = img2[2:, :]
    img2L = img2[:, 1:-1]
    img2R = img2[:, 2:]

    EPI_H = sum(sum(abs(img1R - img1L)))/sum(sum(abs(img2R - img2L)))
    EPI_V = sum(sum(abs(img1U - img1D)))/sum(sum(abs(img2U - img2D)))
    return EPI_H, EPI_V


def calc_gcmse(normed_ref_image, normed_work_image, kappa=0.5, option=1):
    """GCMSE --- Gradient Conduction Mean Square Error.

    Computation of the GCMSE. An image quality assessment measurement 
    for image filtering, focused on edge preservation evaluation. 

    Both input images are compared, returning a float number. As little
    as the GCMSE is, more similar the images are. This metric is edge
    preservation oriented, thus differences between border regions will
    contribute more to the final result.

    The borders are obtained from the reference image, and it only works
    with images of the same scale, size and geometry. This metric is not
    intended to measure any image processing applications but filtering 
    i.e.: it will NOT work for assessing the quality of compression,
    contrast stretching...

    Parameters
    ---------    
    normed_ref_image[]: Array of pixels. Pixel values 0 to 1.
        Reference image. The border regions will be obtained from it. 
        This image is the ideal objective, and the filtered images must
        be as much similar to it as possible.
        
    normed_work_image[]: Array of pixels. Pixel values 0 to 1.
        Image that is compared to the reference one.
        
    kappa: decimal number. Values 0 to 1
        Conductance parameter. It increases the amount of the images
        that are analyzed, as it defines the permisivity for pixels to 
        belong to border regions, and how high is their contribution.
        
    option: integer. Values: 1 or 2
        Select which of the Perona-Malik equations will be used.
        
    Returns
    -------
    gcmse: float
        Value of the GCMSE metric between the 2 provided images. It gets
        smaller as the images are more similar.

    weight: float
        Amount of the image that has been taken into account.     
	"""
    # Initialization and calculation of south and east gradients arrays.
    gradient_S = np.zeros_like(normed_ref_image)
    gradient_E = gradient_S.copy()
    gradient_S[:-1, :] = np.diff(normed_ref_image, axis=0)
    gradient_E[:, :-1] = np.diff(normed_ref_image, axis=1)

    # Image conduction is calculated using the Perona-Malik equations.
    if option == 1:
        cond_S = np.exp(-(gradient_S/kappa) ** 2)
        cond_E = np.exp(-(gradient_E/kappa) ** 2)
    elif option == 2:
        cond_S = 1.0 / (1 + (gradient_S/kappa)**2)
        cond_E = 1.0 / (1 + (gradient_E/kappa)**2)

    # New conduction components are initialized to 1 in order to treat
    # image corners as homogeneous regions
    cond_N = np.ones_like(normed_ref_image)
    cond_W = cond_N.copy()
    # South and East arrays values are moved one position in order to
    # obtain North and West values, respectively.
    cond_N[1:, :] = cond_S[:-1, :]
    cond_W[:, 1:] = cond_E[:, :-1]

    # Conduction module is the mean of the 4 directional values.
    conduction = (cond_N + cond_S + cond_W + cond_E) / 4
    conduction = np.clip(conduction, 0., 1.)
    G = 1 - conduction

    # Calculation of the GCMSE value
    num = ((G*(normed_ref_image - normed_work_image)) ** 2).sum()
    gcmse = num * normed_ref_image.size / G.sum()
    weight = G.sum() / G.size

    return [gcmse, weight]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
