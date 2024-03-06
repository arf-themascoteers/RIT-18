from ann_normal_savi import ANNNormalSAVI
import torch
import torch.nn as nn


class ANNLearnableSAVI(ANNNormalSAVI):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.linear1 = nn.Sequential(
            nn.Linear(1,20),
            nn.LeakyReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(30, 20),
            nn.LeakyReLU(),
        )

        self.linear3 = nn.Sequential(
            nn.Linear(30, 19),
        )

        self.L.requires_grad = True

    def forward(self,x):
        x = self.savi(x)
        x = self.linear1(x)
        l = self.L.repeat(x.shape[0])
        l = l.repeat(10,1)
        l = l.permute(1,0)
        x = torch.cat((x, l), dim=1)
        x = self.linear2(x)
        l = self.L.repeat(x.shape[0])
        l = l.repeat(10,1)
        l = l.permute(1, 0)
        x = torch.cat((x, l), dim=1)
        x = self.linear3(x)
        return x
