import torch
import torch.nn as nn
from ann_simple import ANNSimple
import utils


class ANNNormalSAVI(ANNSimple):
    def __init__(self, train_ds, test_ds, validation_ds, L=0.5):
        super().__init__(train_ds, test_ds, validation_ds)
        self.linear = nn.Sequential(
            nn.Linear(1,20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
            nn.Linear(20,5)
        )
        self.L = nn.Parameter(torch.tensor(L), requires_grad=False)
        self.my_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.my_bias = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self,x):
        x = self.savi(x)
        return x*self.my_weight + self.my_bias

    def savi(self,x):
        red = x[:,2:3]
        nir = x[:,5:6]
        num = (nir - red)*(1+self.L)
        den = (nir+red+self.L)
        return num/den

    def verbose_after(self, ds):
        print(f" L: {round(self.L.item(),4)}", end="")

    def pc(self, ds):
        y = ds.y.numpy()
        return utils.calculate_pc(y, self.savi(ds.x.to(self.device)).reshape(-1).detach().cpu().numpy())

