""" Full assembly of the parts to form the complete network """

from .TopNet_parts import *


class TopNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(TopNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 8))
        self.down1 = (Down(8, 16))
        self.down2 = (Down(16, 32))
        self.down3 = (Down(32, 64))
        self.down4 = (Down(64, 128))

        self.inc1 = (DoubleConv(n_channels, 256))
        self.conv1 = (Conv(64, n_classes))
        self.up1 = (Up(256, 64))
        self.up2 = (Up(64, 32))
        self.up3 = (Up(32, 16))
        self.up4 = (Up(16, 8))

        self.out1 = (OutConv1(2, n_classes))
        self.out2 = (OutConv2(1, n_classes))
        self.out3 = (Conv(8, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.inc1(x5)
        x = self.conv1(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits1 = self.out1(x)

        x = self.inc1(x5)
        x = self.conv1(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits2 = self.out2(x)

        x = self.inc1(x5)
        x = self.conv1(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits3 = self.out3(x)
        return logits3

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)