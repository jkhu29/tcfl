import torch
import torch.nn as nn
import torch.nn.functional as F


class TVLoss(torch.nn.Module):
    """
    TV loss
    """

    def __init__(self):
        super(TVLoss, self).__init__()
 
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2 * (h_tv/count_h + w_tv/count_w) / batch_size
 
    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class GeoLoss(nn.Module):
    def __init__(self, model, criterion=nn.L1Loss()):
        super(GeoLoss, self).__init__()
        self.m = model
        self.criterion = criterion
    
    def forward(self, x, flip_dir="h"):
        out1 = self.m(x)
        out2 = self.geometry_ensemble(x, flip_dir)
        loss = self.criterion(out1, out2)
        return loss

    def geometry_ensemble(self, x, flip_dir="h"):
        if flip_dir == "h":
            flip_axes = [3]
        elif flip_dir == "v":
            flip_axes = [2]
        else:
            raise NotImplementedError

        n, c, h, w = x.shape

        imgs = torch.zeros((8, n, c, h, w)).to(x.device)
        out_imgs = torch.zeros((8, n, c, h, w)).to(x.device)
        for i in range(4):
            imgs[i, ...] = torch.rot90(x, i, [2, 3])
        for i in range(4):
            imgs[4 + i, ...] = torch.flip(imgs[i], flip_axes)

        for i in range(8):
            img = imgs[i, ...]
            temp = self.m(img)
            if i < 4:
                out_imgs[i, ...] = torch.rot90(temp, -i, [2, 3])
            else:
                temp1 = torch.flip(temp, flip_axes)
                out_imgs[i, ...] = torch.rot90(temp1, -(i % 4), [2, 3])

        for i in range(1, 8):
            out_imgs[0] += out_imgs[i]
        return out_imgs[0] / 8


# -----------
# GANLoss: https://github.com/xinntao/BasicSR/blob/master/basicsr/models/losses/losses.py
# -----------
class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.
        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


if __name__ == "__main__":
    from skimage import io
    from models import DnCNN
    from torchvision import transforms
    import matplotlib.pyplot as plt
    model = GeoLoss(DnCNN(1))
    input_numpy = io.imread("./data/Final_Publication_2013_SBSDI_sourcecode_withpcode/test.tif")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(256),
    ])
    input_torch = transform(input_numpy).unsqueeze(0)
    geo_image = model.geometry_ensemble(input_torch)
    plt.subplot(1, 2, 1)
    plt.imshow(input_torch.squeeze().detach().numpy(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(geo_image.squeeze().detach().numpy(), cmap="gray")
    plt.show()
