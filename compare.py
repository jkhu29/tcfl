import torch

noise1 = torch.load("noise1.pkl")
noise2 = torch.load("noise2.pkl")

print(torch.nn.L1Loss()(noise1, noise2))
print(torch.nn.MSELoss()(noise1, noise2))