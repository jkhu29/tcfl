import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=10):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.LeakyReLU(0.2, True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(features, affine=True, track_running_stats=False))
            # layers.append(nn.InstanceNorm2d(features))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        # for noise learning
        # encoder
        self.conv1_1_02 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1_1_02 = nn.BatchNorm2d(64)
        self.conv1_2_02 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2_02 = nn.BatchNorm2d(64)

        self.conv2_1_02 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1_02 = nn.BatchNorm2d(128)
        self.conv2_2_02 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2_02 = nn.BatchNorm2d(128)

        self.conv3_1_02 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1_02 = nn.BatchNorm2d(256)
        self.conv3_2_02 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2_02 = nn.BatchNorm2d(256)

        self.conv4_1_02 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1_02 = nn.BatchNorm2d(512)
        self.conv4_2_02 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2_02 = nn.BatchNorm2d(512)

        self.conv5_1_02 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn5_1_02 = nn.BatchNorm2d(1024)
        self.conv5_2_02 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn5_2_02 = nn.BatchNorm2d(512)

        # decoder
        self.upconv4_1_02 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upbn4_1_02 = nn.BatchNorm2d(512)
        self.upconv4_2_02 = nn.Conv2d(512, 256, 3, padding=1)
        self.upbn4_2_02 = nn.BatchNorm2d(256)

        self.upconv3_1_02 = nn.Conv2d(512, 256, 3, padding=1)
        self.upbn3_1_02 = nn.BatchNorm2d(256)
        self.upconv3_2_02 = nn.Conv2d(256, 128, 3, padding=1)
        self.upbn3_2_02 = nn.BatchNorm2d(128)

        self.upconv2_1_02 = nn.Conv2d(256, 128, 3, padding=1)
        self.upbn2_1_02 = nn.BatchNorm2d(128)
        self.upconv2_2_02 = nn.Conv2d(128, 64, 3, padding=1)
        self.upbn2_2_02 = nn.BatchNorm2d(64)

        self.upconv1_1_02 = nn.Conv2d(128, 32, 3, padding=1)
        self.upbn1_1_02 = nn.BatchNorm2d(32)
        self.upconv1_2_02 = nn.Conv2d(32, 1, 3, padding=1)
        self.upbn1_2_02 = nn.BatchNorm2d(64)

        # ************************************************************
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x0):
        # encoder for noise learning
        x1_1_02 = F.relu(self.bn1_1_02(self.conv1_1_02(x0)))
        x1_2_02 = F.relu(self.bn1_2_02(self.conv1_2_02(x1_1_02)))

        x2_0_02 = self.maxpool(x1_2_02)
        x2_1_02 = F.relu(self.bn2_1_02(self.conv2_1_02(x2_0_02)))
        x2_2_02 = F.relu(self.bn2_2_02(self.conv2_2_02(x2_1_02)))

        x3_0_02 = self.maxpool(x2_2_02)
        x3_1_02 = F.relu(self.bn3_1_02(self.conv3_1_02(x3_0_02)))
        x3_2_02 = F.relu(self.bn3_2_02(self.conv3_2_02(x3_1_02)))

        x4_0_02 = self.maxpool(x3_2_02)
        x4_1_02 = F.relu(self.bn4_1_02(self.conv4_1_02(x4_0_02)))
        x4_2_02 = F.relu(self.bn4_2_02(self.conv4_2_02(x4_1_02)))

        x5_0_02 = self.maxpool(x4_2_02)
        x5_1_02 = F.relu(self.bn5_1_02(self.conv5_1_02(x5_0_02)))
        x5_2_02 = F.relu(self.bn5_2_02(self.conv5_2_02(x5_1_02)))

        # decoder for noise learning
        upx4_1_02 = self.upsample(x5_2_02)
        upx4_2_02 = F.relu(self.upbn4_1_02(self.upconv4_1_02(torch.cat((upx4_1_02, x4_2_02), 1))))
        upx4_3_02 = F.relu(self.upbn4_2_02(self.upconv4_2_02(upx4_2_02)))

        upx3_1_02 = self.upsample(upx4_3_02)
        upx3_2_02 = F.relu(self.upbn3_1_02(self.upconv3_1_02(torch.cat((upx3_1_02, x3_2_02), 1))))
        upx3_3_02 = F.relu(self.upbn3_2_02(self.upconv3_2_02(upx3_2_02)))

        upx2_1_02 = self.upsample(upx3_3_02)
        upx2_2_02 = F.relu(self.upbn2_1_02(self.upconv2_1_02(torch.cat((upx2_1_02, x2_2_02), 1))))
        upx2_3_02 = F.relu(self.upbn2_2_02(self.upconv2_2_02(upx2_2_02)))

        upx1_1_02 = self.upsample(upx2_3_02)
        upx1_2_02 = self.upconv1_1_02(torch.cat((upx1_1_02, x1_2_02), 1))
        noise = self.upconv1_2_02(upx1_2_02)

        return noise


class SingleLayer(nn.Module):
    def __init__(self, inChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.conv = nn.Conv2d(inChannels, growthRate, kernel_size=3, padding=1, bias=True)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class SRDenseNet(nn.Module):
    def __init__(self, growthRate, nDenselayer):
        super(SRDenseNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, growthRate, kernel_size=3, padding=1, bias=True)
        inChannels = growthRate
        
        self.dense1 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer*growthRate

        self.dense2 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense3 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense4 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense5 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense6 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense7 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer*growthRate

        self.dense8 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=256, kernel_size=1,padding=0, bias=True)

        self.conv2 =nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3,padding=1, bias=True)

    def _make_dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels,growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)
                              
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        out = self.dense5(out)
        out = self.dense6(out)
        out = self.dense7(out)
        out = self.dense8(out)
                                         
        out = self.Bottleneck(out)

        out = self.conv2(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


if __name__ == "__main__":
    from torchsummary import summary
    # model = DnCNN(1).cuda()
    # summary(model, (1, 64, 64))
    # model = GeneratorUNet().cuda()
    # summary(model, (1, 64, 64))
    # model = SRDenseNet().cuda()
    # summary(model, (1, 64, 64))
    # model = Discriminator((1, 640, 640)).cuda()
    # summary(model, (1, 640, 640))
