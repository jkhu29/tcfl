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

# seed init
manual_seed = arg.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# dataset init
tcfl_dataset = TCFL_Dataset()
tcfl_dataloader = DataLoader(tcfl_dataset, batch_size=arg.batch_size, shuffle=True)
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

# train
print("-----------------train-----------------")
for epoch in range(arg.niter):
    epoch_losses_gan = utils.AverageMeter()
    epoch_losses_iden = utils.AverageMeter()

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
            ####################
            # BACKWARD PARALLEL
            ####################
            # opt.zero_grad()
            # loss_gan  = trainer._gen_gan_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C)
            # loss_iden = trainer._gen_iden_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C, N_A, N_B)
            # loss_total = loss_gan + arg.lam * loss_iden
            # loss_total.backward()
            # opt.step()

            # trainer.generator.eval()
            # trainer.discrimnator.eval()

            # epoch_losses_gan.update(loss_gan.item(), arg.batch_size)

            ######################
            # BACKWARD NET_D FIRST
            ######################
            # opt_d.zero_grad()
            # for _ in range(arg.critic_updates):
            #     loss_gan = trainer._gen_gan_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C)
            #     epoch_losses_gan.update(loss_gan.item(), arg.batch_size)
            #     loss_gan.backward(retain_graph=True)
            #     opt_d.step()
            # trainer.discrimnator.eval()

            # opt_g.zero_grad()
            # C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, N_A, N_B = trainer.forward(Image_A, Image_B, Clean_C)
            # loss_iden = trainer._gen_iden_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C, N_A, N_B)
            # loss_total = arg.lam * loss_iden + loss_gan
            # loss_total.backward()
            # opt_g.step()
            # trainer.generator.eval()

            ######################
            # BACKWARD NET_G FIRST
            ######################
            loss_iden = trainer._gen_iden_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C, N_A, N_B, sigma=arg.sigma)
            loss_gan = trainer._gen_gan_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C)

            opt_g.zero_grad()
            loss_total = loss_iden * arg.lam + loss_gan
            loss_total.backward()
            opt_g.step()
            trainer.generator.eval()

            opt_d.zero_grad()
            # TODO(Jiakui Hu): sometime it is needed to avoid backward failures
            # C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, N_A, N_B = trainer.forward(Image_A, Image_B, Clean_C)
            for _ in range(arg.critic_updates):
                loss_gan = trainer._gen_gan_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C)
                epoch_losses_gan.update(loss_gan.item(), arg.batch_size)
                loss_gan.backward(retain_graph=True)
                opt_d.step()
            trainer.discrimnator.eval()

            ######################
            # BACKWARD NET_G ONLY
            ######################
            # loss_iden = trainer._gen_iden_loss(C_A1, C_A2, C_B1, C_B2, C_C1, C_C2, Clean_C, N_A, N_B, sigma=arg.sigma)

            # opt_g.zero_grad()
            # loss_iden.backward()
            # opt_g.step()
            # trainer.generator.eval()

            # epoch_losses_gan.update(0, arg.batch_size)

            # print
            epoch_losses_iden.update(loss_iden.item(), arg.batch_size)
            t.set_postfix(
                loss_GAN='{:.6f}'.format(epoch_losses_gan.avg),
                loss_identity='{:.6f}'.format(epoch_losses_iden.avg),
                loss_total='{:.6f}'.format(epoch_losses_gan.avg + arg.lam * epoch_losses_iden.avg)
            )
            t.update(arg.batch_size)
        torch.cuda.empty_cache()

        torch.save(trainer.generator.state_dict(), "./ckpt/generator/epoch_{}.pth".format(epoch+1))
        torch.save(trainer.discrimnator.state_dict(), "./ckpt/discrimnator/epoch_{}.pth".format(epoch+1))
