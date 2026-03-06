import math

import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()

        normalizer_key = "none" if normalization is None else str(normalization).lower()
        if normalizer_key in {"none", "null", "identity", ""}:
            self.normalizer = None
        elif normalizer_key == "batch":
            self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)
        elif normalizer_key == "instance":
            self.normalizer = nn.InstanceNorm1d(embed_dim, affine=True)
        elif normalizer_key == "layer":
            self.normalizer = nn.LayerNorm(embed_dim, elementwise_affine=True)
        else:
            raise ValueError(f"Unsupported normalization type: {normalization}")

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(x)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x
