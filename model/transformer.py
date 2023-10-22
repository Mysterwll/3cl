from torch import nn


class trans(nn.Module):
    def __init__(self):
        super(trans, self).__init__()
        self.trans = nn.Transformer(d_model=531, nhead=9, batch_first=True, dropout=0.0)
        self.line = nn.Linear(531, 2)
        self.softMax = nn.Softmax(dim=2)

    def forward(self, src, tgt):
        x = self.trans(src, tgt)
        x = self.softMax(self.line(x)).reshape(x.shape[0],2)
        return x
