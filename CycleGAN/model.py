import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResnetBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(in_channel),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        net = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        # downsample
        in_channel = 64
        out_channel = in_channel * 2
        for _ in range(2):
            net += [
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel * 2

        # resnet blocks
        for _ in range(9):
            net += [ResnetBlock(in_channel)]

        # upsample
        out_channel = in_channel // 2
        for _ in range(2):
            net += [
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel // 2

        net += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channel, 3, kernel_size=7, padding=0, bias=False),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [nn.Sequential(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        # return self.model(x).view(x.size()[0], -1)


# if __name__ == "__main__":
#     import torch

#     x = torch.randn(1, 3, 256, 256)
#     G = Generator()
#     D = Discriminator()

#     out = G(x)
#     print(out.shape)
#     out = D(out)
#     print(out.shape)
