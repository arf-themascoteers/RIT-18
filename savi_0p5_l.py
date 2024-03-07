from savi_0p5 import SAVI_0p5


class SAVI_0p5_L(SAVI_0p5):
    def __init__(self):
        super().__init__()
        self.L.requires_grad = True
