from evi_2p5_6_7p5_1 import EVI_2p5_6_7p5_1


class EVI_2p5_6_7p5_1_L(EVI_2p5_6_7p5_1):
    def __init__(self):
        super().__init__()
        self.G.requires_grad = True
        self.C1.requires_grad = True
        self.C2.requires_grad = True
        self.L.requires_grad = True

