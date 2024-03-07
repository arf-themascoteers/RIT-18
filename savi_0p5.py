from si import SI
import torch
import torch.nn as nn


class SAVI_0p5(SI):
    def __init__(self):
        super().__init__()
        self.L = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self,x):
        red = x[:,2:3]
        nir = x[:,5:6]
        num = (nir - red)*(1+self.L)
        den = (nir+red+self.L)
        return num/den

