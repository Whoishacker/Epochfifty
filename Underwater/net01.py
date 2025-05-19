import torch
import torch.nn as nn
from torch.nn import Conv2d, LeakyReLU, BatchNorm2d, ConvTranspose2d, ReLU
from lib.ECA import ECA_layer
from lib.CBAM import CBAM
from lib.GAM import GAM
from lib.SENet import SELayer
from torchinfo import summary



# def DepthwiseSeparableConv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
#     layer = nn.Sequential(
#         Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#         Conv2d(in_channels, out_channels, kernel_size=1),
#         BatchNorm2d(out_channels),
#         LeakyReLU(0.2)
#     )
#
#     return layer

def DepthwiseSeparableConv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    layer = nn.Sequential(
        Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
        BatchNorm2d(in_channels),
        ReLU(inplace=True),
        Conv2d(in_channels, out_channels, kernel_size=1),
        BatchNorm2d(out_channels),
        ReLU(inplace=True)
    )

    return layer


def encoder_layer(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    layer = nn.Sequential(
        Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        BatchNorm2d(out_channels),
        LeakyReLU(0.2)
    )
    return layer


def SE_DS_CBAM_Conv(in_channels, out_channels):
    layers = nn.Sequential(
        SELayer(in_channels, in_channels),
        encoder_layer(in_channels, out_channels)
    )
    return layers

def decoder_layer(in_channels, out_channels, last_layer=False, kernel_size=4, stride=2, padding=1):
    if not last_layer:
        layer = nn.Sequential(
            ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.ReLU()

        )
    else:
        layer = nn.Sequential(
            ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )
    return layer



def discrimiter_layer(in_channels, out_channels, kernel_size=4, stride=2, padding=1, wgan=False):
    if wgan:
        layer = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(out_channels),
            LeakyReLU(0.2)
        )
    else:
        layer = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            LeakyReLU(0.2)
        )
    return layer


class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        # Encoder
        self.enc_conv1 = SE_DS_CBAM_Conv(3, 32)
        self.cbam1 = CBAM(32)
        self.enc_conv2 = SE_DS_CBAM_Conv(32, 64)
        self.cbam2 = CBAM(64)
        self.enc_conv3 = SE_DS_CBAM_Conv(64, 128)
        self.cbam3 = CBAM(128)
        self.enc_conv4 = SE_DS_CBAM_Conv(128, 256)
        self.cbam4 = CBAM(256)
        self.enc_conv5 = SE_DS_CBAM_Conv(256, 256)
        self.cbam5 = CBAM(256)

        # Decoder
        self.dec_conv1 = decoder_layer(256, 256)
        self.dec_conv2 = decoder_layer(512, 128)
        self.dec_conv3 = decoder_layer(256, 64)
        self.dec_conv4 = decoder_layer(128, 32)
        self.dec_conv5 = decoder_layer(64, 3, last_layer=True)


    def forward(self, input_x):
        # Encoder
        output_enc_conv1 = self.enc_conv1(input_x)
        output_enc_conv1_cbam = self.cbam1(output_enc_conv1)

        output_enc_conv2 = self.enc_conv2(output_enc_conv1_cbam)
        output_enc_conv2_cbam = self.cbam2(output_enc_conv2)

        output_enc_conv3 = self.enc_conv3(output_enc_conv2_cbam)
        output_enc_conv3_cbam = self.cbam3(output_enc_conv3)

        output_enc_conv4 = self.enc_conv4(output_enc_conv3_cbam)
        output_enc_conv4_cbam = self.cbam4(output_enc_conv4)

        output_enc_conv5 = self.enc_conv5(output_enc_conv4_cbam)
        output_enc_conv5_cbam = self.cbam5(output_enc_conv5)

        #  Decoder
        output_dec_conv1 = self.dec_conv1(output_enc_conv5_cbam)
        output_dec_conv1 = torch.cat([output_dec_conv1, output_enc_conv4], dim=1)

        output_dec_conv2 = self.dec_conv2(output_dec_conv1)
        output_dec_conv2 = torch.cat([output_dec_conv2, output_enc_conv3], dim=1)

        output_dec_conv3 = self.dec_conv3(output_dec_conv2)
        output_dec_conv3 = torch.cat([output_dec_conv3, output_enc_conv2], dim=1)

        output_dec_conv4 = self.dec_conv4(output_dec_conv3)
        output_dec_conv4 = torch.cat([output_dec_conv4, output_enc_conv1], dim=1)

        output_dec_conv5 = self.dec_conv5(output_dec_conv4)


        return output_dec_conv5



class DiscrimiterNet(torch.nn.Module):
    def __init__(self, wgan_loss):
        super(DiscrimiterNet, self).__init__()
        self.wgan_loss = wgan_loss

        self.conv1 = discrimiter_layer(3, 64, self.wgan_loss)
        self.conv2 = discrimiter_layer(64, 128, self.wgan_loss)
        self.conv3 = discrimiter_layer(128, 256, self.wgan_loss)
        self.conv4 = discrimiter_layer(256, 512, self.wgan_loss)
        self.conv5 = discrimiter_layer(512, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x




if __name__=='__main__':
    import datetime
    a = datetime.datetime.now()
    # 创建模型实例
    netG = GeneratorNet()

    # 生成随机输入数据
    input_x = torch.randn(32, 3, 256, 256)

    # 前向传播
    output = netG(input_x)
    summary(netG, (32, 3, 256, 256))

    print(f"Output shape: {output.shape}")
    b = datetime.datetime.now()
    print(b-a)