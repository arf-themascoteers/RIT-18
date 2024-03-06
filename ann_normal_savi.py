import torch
import torch.nn as nn
from ann_simple import ANNSimple


class ANNNormalSAVI(ANNSimple):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.linear = nn.Sequential(
            nn.Linear(1,20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20,19)
        )
        self.L = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self,x):
        x = self.savi(x)
        return self.linear(x)

    def savi(self,x):
        red = x[:,2:3]
        nir = x[:,5:6]
        num = (nir - red)*(1+self.L)
        den = (nir+red+self.L)
        return num/den

    def verbose_after(self, ds):
        print(" L:",round(self.L.item(),4), end="")



