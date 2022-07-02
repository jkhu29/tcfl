import argparse


def get_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers, you had better put it '
                                                               '4 times of your gpu')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size, default=64')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for, default=10')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='select the learning rate, default=3e-5')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='select the learning rate, default=3e-5')
    parser.add_argument('--seed', type=int, default=118, help="random seed")
    parser.add_argument('--lam', type=float, default=6, help="lam * loss_iden + loss_gan")
    parser.add_argument('--critic_updates', type=int, default=1, help="more backward for NET_D")
    parser.add_argument('--sigma', type=float, default=-0.2, help="constructive loss")
    opt = parser.parse_args()
    return opt
