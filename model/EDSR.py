import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.sub_blocks import ResidualBlock, UpsampleBLock, MeanShift


class EDSR(nn.Module):
    """Generator of SRGAN.

    Args:
        scale_factor (int): scale factor of target
        B (int): number of residual blocks
    """
    def __init__(self, scale_factor: int = 2, B: int = 32, channels: int = 256,
                 in_channels: int = 3, out_channels: int = 3, rgb_range: int = 255) -> None:
        super(EDSR, self).__init__()
        self.sub_mean = MeanShift(rgb_range)
        
        # first layer
        self.block1 = nn.Conv2d(in_channels, channels, kernel_size=9, stride=1, padding=4)
        
        # residual blocks
        residual_blocks = [ResidualBlock(channels) for _ in range(B)]
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # second conv layer after residual blocks
        self.block2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        # upscale block
        upscale_block = [UpsampleBLock(channels, scale_factor),
                         nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)]
        self.upscale_block = nn.Sequential(*upscale_block)
        
        self.add_mean = MeanShift(rgb_range, sign=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.sub_mean(x)
        out2 = self.block1(out1)
        out3 = self.residual_blocks(out2)
        out4 = self.block2(out3)
        out4 = torch.add(out4, out2)
        out5 = self.upscale_block(out4)
        out6 = self.add_mean(out5)
        
        return out6


if __name__ == "__main__":
    from torchsummary import summary
    model = EDSR()
    summary(model, (3, 224, 224), device="cpu")