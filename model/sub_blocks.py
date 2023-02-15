import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        
        for param in self.parameters():
            param.requires_grad = False
            

class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        
        return torch.add(out, identity)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        block_layers = []
        if up_scale == 3:
            block_layers.append(nn.Conv2d(in_channels, in_channels * 9, kernel_size=3, padding=1))
            block_layers.append(nn.PixelShuffle(3))
            block_layers.append(nn.PReLU())
        elif (up_scale & (up_scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(up_scale, 2))):
                block_layers.append(nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1))
                block_layers.append(nn.PixelShuffle(2))
                block_layers.append(nn.PReLU())
        else:
            raise NotImplementedError
        
        self.block_layers = nn.Sequential(*block_layers)

    def forward(self, x):
        return self.block_layers(x)


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)  
    m = MeanShift(sign=-1)(x)