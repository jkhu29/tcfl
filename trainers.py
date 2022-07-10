import torch
import torch.nn as nn

import utils
import models
import criterions


class TCFL_Trainer(object):
    def __init__(self, in_channels: int = 3, device="cuda", patch_size: int = 640):
        super().__init__()
        # models
        self.generator = models.DnCNN(channels=in_channels).to(device)
        # self.generator = models.GeneratorUNet().to(device)
        self.discrimnator = models.Discriminator((in_channels, patch_size, patch_size)).to(device)
        # self.generator.apply(utils.weights_init)
        # self.discrimnator.apply(utils.weights_init)
        # criterions
        self.criterion_identity = nn.L1Loss().to(device)
        self.criterion_GAN = criterions.GANLoss("lsgan").to(device)
        self.criterion_TV = criterions.TVLoss()

    def forward(self, Image_A: torch.Tensor, Image_B: torch.Tensor, C_C: torch.Tensor):
        """
        I: Image
        C: Clean Component
        N: Noise
        I = C + N

        generator: I --> N
        """
        # N2P Part
        N_A = self.generator(Image_A)
        N_B = self.generator(Image_B)
        C_A1 = Image_A - N_A
        C_B1 = Image_B - N_B

        # P2P Part
        I_A = C_A1 + N_B
        I_B = C_B1 + N_A
        C_A2 = I_A - self.generator(I_A)
        C_B2 = I_B - self.generator(I_B)

        # C2P Part
        I_C1 = C_C + N_A
        I_C2 = C_C + N_B
        C_C1 = I_C1 - self.generator(I_C1)
        C_C2 = I_C2 - self.generator(I_C2)

        return C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, N_A, N_B

    def _gen_iden_loss(self, C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, C_C, N_A, N_B, epoch, mu: float = 1.0, sigma: float = 0.0):
        Loss_A = self.criterion_identity(C_A1, C_A2)
        Loss_B = self.criterion_identity(C_B1, C_B2)
        Loss_C1 = self.criterion_identity(C_C1, C_C)
        Loss_C2 = self.criterion_identity(C_C2, C_C)
        # Loss_N = self.criterion_identity(N_A, N_B)
        # print(C_A1.max().item(), C_A2.max().item())
        # print(Loss_A, Loss_B, Loss_C1, Loss_C2, Loss_N)
        return Loss_A + Loss_B + mu * (Loss_C1 + Loss_C2) + sigma * (
            self.criterion_TV(C_A1) + \
            self.criterion_TV(C_A2) + \
            self.criterion_TV(C_B1) + \
            self.criterion_TV(C_B2) + \
            self.criterion_TV(C_C1) + \
            self.criterion_TV(C_C2)
        )

    def _gen_gan_loss(self, C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, C_C, forD: bool = False, wgan: bool = False):
        target_C_C = self.discrimnator(C_C)
        target_C_A1 = self.discrimnator(C_A1)
        target_C_A2 = self.discrimnator(C_A2)
        target_C_B1 = self.discrimnator(C_B1)
        target_C_B2 = self.discrimnator(C_B2)
        target_C_C1 = self.discrimnator(C_C1)
        target_C_C2 = self.discrimnator(C_C2)

        if not wgan:
            Loss_C = self.criterion_GAN(target_C_C, True, True)
            Loss_A1 = self.criterion_GAN(target_C_A1, False, True)
            Loss_A2 = self.criterion_GAN(target_C_A2, False, True)
            Loss_B1 = self.criterion_GAN(target_C_B1, False, True)
            Loss_B2 = self.criterion_GAN(target_C_B2, False, True)
            Loss_C1 = self.criterion_GAN(target_C_C1, False, True)
            Loss_C2 = self.criterion_GAN(target_C_C2, False, True)

            if forD:
                return (Loss_A1 + Loss_A2 + \
                    Loss_B1 + Loss_B2 + \
                    Loss_C1 + Loss_C2) / 6 + Loss_C
            else:
                return Loss_A1 + Loss_A2 + \
                    Loss_B1 + Loss_B2 + \
                    Loss_C1 + Loss_C2
        else:
            if forD:
                return (target_C_A1.mean() + target_C_A2.mean() + \
                    target_C_B1.mean() + target_C_B2.mean() + \
                    target_C_C1.mean() + target_C_C2.mean()) / 6 - target_C_C.mean()
            else:
                return target_C_A1.mean() + target_C_A2.mean() + \
                    target_C_B1.mean() + target_C_B2.mean() + \
                    target_C_C1.mean() + target_C_C2.mean()

    def _gen_loss(self, C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, C_C, mu: float = 1.0, sigma: float = 0.0, lam: float = 6.0):
        loss_iden = self._gen_iden_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, C_C, mu=mu, sigma=sigma)
        loss_gan = self._gen_gan_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, C_C)
        loss_total = loss_gan + lam * loss_iden
        return loss_total


if __name__ == "__main__":
    trainer = TCFL_Trainer(1)
    dummy_A = torch.rand((2, 1, 64, 64)).cuda()
    dummy_B = torch.rand((2, 1, 64, 64)).cuda()
    dummy_C = torch.rand((2, 1, 64, 64)).cuda()
    loss = trainer.forward(dummy_A, dummy_B, dummy_C)
    print(loss)
