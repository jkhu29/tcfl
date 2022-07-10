import os
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import trainers
import config
import utils
from dataset import TCFL_Dataset


# torch.autograd.set_detect_anomaly(True)

arg = config.get_options()

# deveice init
CUDA_ENABLE = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA_ENABLE else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# seed init
manual_seed = arg.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# dataset init
tcfl_dataset = TCFL_Dataset()
tcfl_dataloader = DataLoader(tcfl_dataset, batch_size=arg.batch_size, shuffle=True, pin_memory=True, drop_last=True)
# valid_dataset = TFRecordDataset(arg.valid_file, None, description)
# valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)
length = len(tcfl_dataset)

# models init
trainer = trainers.TCFL_Trainer(in_channels=1, device=device)

# optim and scheduler init
# opt = optim.Adam(
    # itertools.chain(trainer.generator.parameters(), trainer.discrimnator.parameters()),
    # lr=arg.lr_g, betas=[0.5, 0.999]
# )
opt_g = optim.Adam(trainer.generator.parameters(), lr=arg.lr_g, betas=[0.5, 0.999])
opt_d = optim.Adam(trainer.discrimnator.parameters(), lr=arg.lr_d, betas=[0.5, 0.999])

# Buffers of previously generated samples
C_A1_buffer = utils.ReplayBuffer()
C_A2_buffer = utils.ReplayBuffer()
C_B1_buffer = utils.ReplayBuffer()
C_B2_buffer = utils.ReplayBuffer()
C_C1_buffer = utils.ReplayBuffer()
C_C2_buffer = utils.ReplayBuffer()

# train
print("-----------------train-----------------")
for epoch in range(arg.niter):
    epoch_losses_gan = utils.AverageMeter()
    epoch_losses_iden = utils.AverageMeter()
    epoch_losses_total = utils.AverageMeter()

    with tqdm(total=(length - length % arg.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch + 1, arg.niter))

        for Image_A, Image_B, Clean_C in tcfl_dataloader:
            # Image_A, Image_B, Clean_C = record
            Image_A = Image_A.to(device)
            Image_B = Image_B.to(device)
            Clean_C = Clean_C.to(device)

            trainer.generator.train()
            trainer.discrimnator.train()
            C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, N_A, N_B = trainer.forward(Image_A, Image_B, Clean_C)

            ######################
            # BACKWARD NET_G FIRST
            ######################
            loss_iden = trainer._gen_iden_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C, N_A, N_B, epoch=epoch, sigma=arg.sigma)
            loss_gan = trainer._gen_gan_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C, forD=False, wgan=True)

            opt_g.zero_grad()
            loss_total = loss_iden * arg.lam - loss_gan
            loss_total.backward()
            opt_g.step()

            opt_d.zero_grad()
            for _ in range(arg.critic_updates):
                C_A1_ = C_A1_buffer.push_and_pop(C_A1)
                C_A2_ = C_A2_buffer.push_and_pop(C_A2)
                C_B1_ = C_B1_buffer.push_and_pop(C_B1)
                C_B2_ = C_B2_buffer.push_and_pop(C_B2)
                C_C1_ = C_C1_buffer.push_and_pop(C_C1)
                C_C2_ = C_C2_buffer.push_and_pop(C_C2)
                loss_gan_ = trainer._gen_gan_loss(
                    C_A1_.detach(), C_A2_.detach(), 
                    C_B1_.detach(), C_B2_.detach(), 
                    C_C1_.detach(), C_C2_.detach(), 
                    Clean_C, forD=True, wgan=True
                )
                epoch_losses_gan.update(loss_gan_.item(), arg.batch_size)
                loss_gan_.backward()
                opt_d.step()

                # param limit
                for p in trainer.discrimnator.parameters():
                    p.data.clamp(-0.001, 0.001)

            # print
            epoch_losses_iden.update(loss_iden.item(), arg.batch_size)
            epoch_losses_total.update(loss_total.item(), arg.batch_size)
            t.set_postfix(
                loss_GAN='{:.6f}'.format(epoch_losses_gan.avg),
                loss_identity='{:.6f}'.format(epoch_losses_iden.avg),
                loss_total='{:.6f}'.format(epoch_losses_total.avg)
            )
            t.update(arg.batch_size)
        torch.cuda.empty_cache()

        torch.save(trainer.generator.state_dict(), "./ckpt/generator/epoch_{}.pth".format(epoch+1))
        torch.save(trainer.discrimnator.state_dict(), "./ckpt/discrimnator/epoch_{}.pth".format(epoch+1))
