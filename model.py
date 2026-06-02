
import torch
import torch.nn as nn

class UIR_PolyKernel(nn.Module):
    def __init__(self):
        super(UIR_PolyKernel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=31, padding=15),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
