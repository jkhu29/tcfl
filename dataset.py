import os
import copy
import random
from PIL import Image

from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TCFL_Dataset(Dataset):
    def __init__(self, dataset_name: str = "PKU37", transform = None) -> None:
        super().__init__()
        self.clean_images_paths = []
        self.noise_images_paths = []
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.dataset_name = dataset_name
        # init
        self.__get_images_paths()

    def __get_images_paths(self) -> None:
        if self.dataset_name == "PKU37":
            self.clean_images_path = "./data/PKU37_OCT_Denoising/PKU37_OCT_Denoising/clean/"
            self.noise_images_path = "./data/PKU37_OCT_Denoising/PKU37_OCT_Denoising/noisy/"
            self.clean_images_paths = os.listdir(self.clean_images_path)
            self.noise1_images_paths = os.listdir(self.noise_images_path)
            self.noise2_images_paths = copy.deepcopy(self.noise1_images_paths)
            random.shuffle(self.noise2_images_paths)

    def __len__(self):
        return len(self.noise1_images_paths)

    def __getitem__(self, index):
        if self.dataset_name == "PKU37":
            # I_A, I_B, C_C = self.images_paths[index]
            I_A = self.noise1_images_paths[index]
            I_B = self.noise2_images_paths[index]
            C_C = self.clean_images_paths[index % len(self.clean_images_paths)]
            I_A_path = os.path.join(self.noise_images_path, I_A)
            I_B_path = os.path.join(self.noise_images_path, I_B)
            C_C_path = os.path.join(self.clean_images_path, C_C)

            I_A_numpy = io.imread(I_A_path)
            I_B_numpy = io.imread(I_B_path)
            C_C_numpy = io.imread(C_C_path)

            I_A_image = Image.fromarray(I_A_numpy)
            I_B_image = Image.fromarray(I_B_numpy)
            C_C_image = Image.fromarray(C_C_numpy)

            I_A_torch = self.transform(I_A_image).float()
            I_B_torch = self.transform(I_B_image).float()
            C_C_torch = self.transform(C_C_image).float()
        return I_A_torch, I_B_torch, C_C_torch


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    tcfl_dataset = TCFL_Dataset()
    tcfl_dataloader = DataLoader(tcfl_dataset, batch_size=1, shuffle=True)
    print(len(tcfl_dataset))

    for I_A, I_B, C_C in tcfl_dataloader:
        plt.subplot(1, 3, 1)
        plt.imshow(I_A.squeeze().numpy(), cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow(I_B.squeeze().numpy(), cmap="gray")
        plt.subplot(1, 3, 3)
        plt.imshow(C_C.squeeze().numpy(), cmap="gray")
        plt.show()
        break
