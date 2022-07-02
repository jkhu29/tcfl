# TCFL

train:

```shell
# PKU37 --> ./data/
python train.py --batch_size 2
```

test:

```shell
# modify the path of image in test.py
python test.py
```

> EPI, GCMSE, PSNR, SSIM 等指标的计算均在 utils.py 中

传统方法:

```shell
python BM3D.py
python NLM.py
```

> 传统方法结果均在./img/下

## 问题

我觉得 TCFL 有一些问题

### 数据

数据应该是只公开了一部分，论文里面是1850个噪声图像，实际是1732，应该是只有训练集

### 判别器模型结构

在 Fig.4 中，判别器结构应当如下：

```shell
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 320, 320]           1,088
         LeakyReLU-2         [-1, 64, 320, 320]               0
            Conv2d-3        [-1, 128, 160, 160]         131,200
    InstanceNorm2d-4        [-1, 128, 160, 160]               0
         LeakyReLU-5        [-1, 128, 160, 160]               0
            Conv2d-6          [-1, 256, 80, 80]         524,544
    InstanceNorm2d-7          [-1, 256, 80, 80]               0
         LeakyReLU-8          [-1, 256, 80, 80]               0
            Conv2d-9          [-1, 512, 40, 40]       2,097,664
   InstanceNorm2d-10          [-1, 512, 40, 40]               0
        LeakyReLU-11          [-1, 512, 40, 40]               0
           Conv2d-12          [-1, 512, 39, 39]       4,194,816
  ReflectionPad2d-13          [-1, 512, 40, 40]               0
   InstanceNorm2d-14          [-1, 512, 40, 40]               0
        LeakyReLU-15          [-1, 512, 40, 40]               0
           Conv2d-16            [-1, 1, 39, 39]           8,193
  ReflectionPad2d-17            [-1, 1, 40, 40]               0
================================================================
Total params: 6,957,505
Trainable params: 6,957,505
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.56
Forward/backward pass size (MB): 255.97
Params size (MB): 26.54
Estimated Total Size (MB): 284.07
----------------------------------------------------------------
```

也就是后面几层是 40 \* 40 的分辨率

但是原始的 PatchGAN 论文提供的[代码](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1/models/networks.py#L538)中的网络结构是

```shell
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 320, 320]           1,088
         LeakyReLU-2         [-1, 64, 320, 320]               0
            Conv2d-3        [-1, 128, 160, 160]         131,200
    InstanceNorm2d-4        [-1, 128, 160, 160]               0
         LeakyReLU-5        [-1, 128, 160, 160]               0
            Conv2d-6          [-1, 256, 80, 80]         524,544
    InstanceNorm2d-7          [-1, 256, 80, 80]               0
         LeakyReLU-8          [-1, 256, 80, 80]               0
            Conv2d-9          [-1, 512, 79, 79]       2,097,664
   InstanceNorm2d-10          [-1, 512, 79, 79]               0
        LeakyReLU-11          [-1, 512, 79, 79]               0
           Conv2d-12            [-1, 1, 78, 78]           8,193
================================================================
Total params: 2,762,689
Trainable params: 2,762,689
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.56
Forward/backward pass size (MB): 285.68
Params size (MB): 10.54
Estimated Total Size (MB): 297.78
----------------------------------------------------------------
```

后面几层是没有降分辨率的（由于 kernel 是4所以会分辨率会减1）

**判别器在这种训练方式中很重要，结构可能并不是这个样子，loss_GAN的计算可能也有些许问题**

### 训练方式

```python
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
```

```self.generator(X) = Y``` 中 X 代表输入带噪声的图片， Y 代表输出的噪声。仅使用上述的训练方式会导致一个问题：在任意输入的时候 generator 输出一个固定的张量 A 即可。

我们假定 ```self.generator(x) = A```， 其中 x 可以是任一张量， A 是一个确定的张量。带入数据流：

```python
# N2P Part
N_A = A
N_B = A
C_A1 = Image_A - A
C_B1 = Image_B - A

# P2P Part
I_A = C_A1 + N_A = C_A1 + A = Image_A
I_B = C_B1 + N_A = C_B1 + A = Image_B
C_A2 = I_A - A = Image_A
C_B2 = I_B - A = Image_B

# C2P Part
I_C1 = C_C + N_A = C_C + A
I_C2 = C_C + N_B = C_C + A
C_C1 = I_C1 - self.generator(I_C1) = I_C1 - A = C_C
C_C2 = I_C2 - self.generator(I_C2) = I_C2 = A = C_C
```

我们的损失函数中有关图像质量的损失构成如下：

```shell
Loss_A = L1Loss()(C_A1, C_A2)
Loss_B = L1Loss()(C_B1, C_B2)
Loss_C1 = L1Loss()(C_C1, C_C)
Loss_C2 = L1Loss()(C_C2, C_C)
```

我们带入假定 ```self.generator(x) = A``` 的结果，可以发现

```shell
Loss_A = L1Loss()(Image_A - A, Image_A)
Loss_B = L1Loss()(Image_B - A, Image_B)
Loss_C1 = L1Loss()(C_C, C_C) = 0
Loss_C2 = L1Loss()(C_C, C_C) = 0
```

此时 A $\to$ 0 的时候，整个损失就为0

很显然这种训练方法（不加入判别器损失的时候）极容易陷入这种局部最优。

**由于算力不足，我测试了几次小分辨率下训练的情况，小分辨率下加入了判别器的损失也很容易直接生成器只会生成0。在只有少量数据的情况下也会有类似问题。在一般的情况（加入了判别器损失）下，会生成一个a * E(单位矩阵)， a是一个小于1的数，为什么会发生这种情况我也没有很理解（感觉也可能因为显存不够我设置的batch_size是1）。**

#### 关于生成器只能生成固定矩阵的举例

![](./img/epoch_100_1.png)

![](./img/epoch_100_2.png)

上述两者生成的 noise 张量的差距为

| MSE | MAE |
| -- | -- |
| 5.1258e-17 | 9.1298e-10 |

```noise1.pkl``` 与 ```noise2.pkl``` 均是由 ```./ckpt/generator/epoch_100.pth``` 生成的结果

```shell
python compare.py
```

## 可以只用一个生成器嘛

判别器是用来判断 Clean 的图像是人为创造的还是原始的，个人认为它还可以避免陷入上述说的局部最优问题。但是拉开 N_A 与 N_B 的距离理论上也能防止生成器只能生成固定矩阵的结果。这也意味着或许可能只用一个生成器就能训练。

见 ```./trainers.py#L56``` 引入 ```sigma * (exp(-Loss(N_A, N_B)) - 1)``` 后，训练结果如下：

| 0.1 | 0.5 | 1 |
| -- | -- | -- |
| ![](./img/0.1.png) | ![](./img/0.5.png) | ![](./img/1.png) |
| 2.5 | 5 | 10 |
| ![](./img/2.5.png) | ![](./img/5.png) | ![](./img/10.png) |

sigma 大于1的时候确实拉开了 N_A 与 N_B，但是拉的太开了...

而在 sigma 小于1的时候跟上面包含判别器的效果比较一致

> 直接在原质量损失上减去 Loss(N_A, N_B) 也有用

**可能可以通过调参找到一个参数能够做到单一生成器训练，但是极不稳定。个人感觉我的复现在判别器及其损失的实现上应该有比较大的问题**
