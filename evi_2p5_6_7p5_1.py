import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, root_mean_squared_error
import utils


class EVI_2p5_6_7p5_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = nn.Parameter(torch.tensor(2.5), requires_grad=False)
        self.C1 = nn.Parameter(torch.tensor(6), requires_grad=False)
        self.C2 = nn.Parameter(torch.tensor(7.5), requires_grad=False)
        self.L = nn.Parameter(torch.tensor(1), requires_grad=False)

    def forward(self,x):
        blue = x[:,0:1]
        red = x[:,2:3]
        nir = x[:,5:6]
        num = (nir - red)
        den = nir + (self.C1*red) - (self.C2*blue) + self.L
        return num/den

