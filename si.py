import torch
import torch.nn as nn


class SI(nn.Module):
    def __init__(self):
        super().__init__()

    def params_dic(self):
        dic = {}
        for name, param in self.named_parameters():
            if param.numel() == 1:
                dic[name] = param.item()

