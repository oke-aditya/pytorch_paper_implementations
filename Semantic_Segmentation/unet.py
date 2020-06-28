import torch
import torch.nn as nn

# This is implementation of orignal U-Net paper
# https://arxiv.org/pdf/1505.04597.pdf
# Thanks to Abhishek Thakur for his video and implementation.
# https://www.youtube.com/watch?v=u1loyDCoGbE
# This is just a re-implementation of his work.


def double_conv_block(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)
        ),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(stride=2),
        nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3)
        ),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(stride=2),
    )
    return conv


def crop_img(input_tensor, target):
    # Since we have square images
    target_size = target.size()[2]
    input_tensor_size = input_tensor.size()[2]
    delta = input_tensor_size - target_size
    delta = delta // 2
    return input_tensor[
        :, :, delta : input_tensor_size - delta, delta : input_tensor_size - delta
    ]


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv_1 = double_conv_block(1, 64)
        self.double_conv_2 = double_conv_block(64, 128)
        self.double_conv_3 = double_conv_block(128, 256)
        self.double_conv_4 = double_conv_block(256, 512)
        self.double_conv_5 = double_conv_block(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2
        )
        self.up_conv_1 = double_conv_block(in_channels=1024, out_channels=512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2
        )
        self.up_conv_2 = double_conv_block(in_channels=512, out_channels=256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2
        )
        self.up_conv_3 = double_conv_block(in_channels=256, out_channels=128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2
        )
        self.up_conv_4 = double_conv_block(in_channels=128, out_channels=64)

        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))

    def forward(self, x):
        # Encoder Part
        x1 = self.double_conv_1(x)  #
        # print(x1.shape)
        x2 = self.max_pool_2x2(x1)
        # print(x2.shape)

        x3 = self.double_conv_2(x2)  #
        # print(x3.shape)
        x4 = self.max_pool_2x2(x3)

        x5 = self.double_conv_3(x4)  #
        # print(x5.shape)
        x6 = self.max_pool_2x2(x5)

        x7 = self.double_conv_4(x6)  #
        # print(x7.shape)
        x8 = self.max_pool_2x2(x7)

        x9 = self.double_conv_5(x8)
        # x = self.max_pool_2x2(x)
        # print(x9.shape)

        x = self.up_trans_1(x9)
        # print(x.shape)
        y = crop_img(x7, x)
        # print(y.shape)
        x = self.up_conv_1(torch.cat([x, y], 1))
        # print(x.shape)

        x = self.up_trans_2(x)
        # print(x.shape)
        y = crop_img(x5, x)
        # print(y.shape)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        # print(x.shape)
        y = crop_img(x3, x)
        # print(y.shape)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        # print(x.shape)
        y = crop_img(x1, x)
        # print(y.shape)
        x = self.up_conv_4(torch.cat([x, y], 1))
        # print(x.shape)

        x = self.out(x)
        # print(x.shape)

        return x


if __name__ == "__main__":
    net = Unet()
    image = torch.rand((1, 1, 572, 572))
    out = net(image)
