import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, root_mean_squared_error
import utils


class SAVI_0p5(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self,x):
        red = x[:,2:3]
        nir = x[:,5:6]
        num = (nir - red)*(1+self.L)
        den = (nir+red+self.L)
        return num/den

