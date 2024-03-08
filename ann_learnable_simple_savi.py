from ann_normal_savi import ANNNormalSAVI
import torch
import torch.nn as nn


class ANNLearnableSimpleSAVI(ANNNormalSAVI):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.L.requires_grad = True

