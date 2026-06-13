import torch
import torch.nn as nn

NUM_ITERS = 8

class DCENet(nn.Module):
    def __init__(self, num_channels: int = 32, num_iters: int = NUM_ITERS):
        super().__init__()
        self.num_iters = num_iters

        self.conv1 = nn.Conv2d(3, num_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)

        self.conv5 = nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1)
        self.conv6 = nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1)
        self.conv7 = nn.Conv2d(num_channels * 2, num_iters * 3, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        c3 = self.relu(self.conv3(c2))
        c4 = self.relu(self.conv4(c3))

        c5 = self.relu(self.conv5(torch.cat([c4, c3], dim=1)))
        c6 = self.relu(self.conv6(torch.cat([c5, c2], dim=1)))
        x_r = self.tanh(self.conv7(torch.cat([c6, c1], dim=1)))
        return x_r

def apply_curves(image: torch.Tensor, curve_maps: torch.Tensor, num_iters: int = NUM_ITERS) -> torch.Tensor:
    x = image
    for i in range(num_iters):
        r = curve_maps[:, i*3:(i+1)*3, :, :]   
        x = x + r * (torch.pow(x, 2) - x)
    return x