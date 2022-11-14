import torch
from torch import nn
from torch.nn import functional as F


class ParallelNets(nn.Module):
    def __init__(self, in_ch: int=2, out_ch: int=1, init_features: int=64, dropout_prob: float=0):
        """UNet segmentation model with a parallel FCNN regressor for the tip position.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels
            init_features: number of initial features
            dropout_prob: between 0 and 1 of dropout probability

        """
        super().__init__()

        self.unet = UNet(in_ch=in_ch, out_ch=out_ch, init_features=init_features,
                         dropout_prob=dropout_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(in_features=init_features*8, out_features=1024),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x1 = self.unet.inc(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)

        x5 = self.unet.base(x5)

        x = self.unet.up1(x5, x4)
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        x = self.unet.outc(x)

        y = self.avgpool(x5)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        y = self.out(y)
        return torch.sigmoid(x), y


class UNet(nn.Module):
    def __init__(self, in_ch: int=2, out_ch: int=1, init_features: int=64, dropout_prob: float=0):
        """UNet with 4 blocks like originally proposed by Ronneberger et al.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels
            init_features: number of initial features
            dropout_prob: between 0 and 1 of dropout probability

        """
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.init_features = init_features
        self.dropout_prob = dropout_prob

        self.inc = DoubleConv(self.in_ch, self.init_features)
        self.down1 = Down(self.init_features, self.init_features*2)
        self.down2 = Down(self.init_features*2, self.init_features*4)
        self.down3 = Down(self.init_features*4, self.init_features*8)
        self.down4 = Down(self.init_features*8, self.init_features*8)

        self.base = Base(self.init_features*8, dropout_prob=dropout_prob)

        self.up1 = Up(self.init_features*16, self.init_features*4)
        self.up2 = Up(self.init_features*8, self.init_features*2)
        self.up3 = Up(self.init_features*4, self.init_features)
        self.up4 = Up(self.init_features*2, self.init_features)
        self.outc = OutConv(self.init_features, self.out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.base(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """(Conv => BN => ReLU) * 2

        Args:
            in_ch: number of input channels
            out_ch: number of output channels

        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """MaxPool => DoubleConv

        Args:
            in_ch: number of input channels
            out_ch: number of output channels

        """
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doubleconv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.doubleconv(x)
        return x


class Base(nn.Module):
    def __init__(self, channels: int, dropout_prob: float):
        """(Dropout => Conv => BN => LeakyReLU) * 2

        Args:
            channels: number of input channels
            dropout_prob: between 0 and 1 of dropout probability

        """
        super().__init__()

        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool=False):
        """Upsample or ConvTrans2D => DoubleConv

        Args:
            in_ch: number of input channels
            out_ch: number of output channels
            bilinear: if True, then bilinear Upsampling is used
                      if False, then Transposed Convolutions are used
        """

        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.doubleconv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.doubleconv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """Final convolutional layer.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels

        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
