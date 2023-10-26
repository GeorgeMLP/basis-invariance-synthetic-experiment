import torch
import torch.nn.functional as F
from typing import Callable


class DeepSets(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: bool, dropout: float, width1: int, width2: int,
                 activation: Callable, readout: str = "mean", random_sign: bool = False):
        super(DeepSets, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, width1)
        self.lin2 = torch.nn.Linear(width1, width2)
        self.classifier = torch.nn.Linear(width2, out_channels)
        self.layernorm = torch.nn.LayerNorm(width1, eps=1e-6)
        self.activation = activation
        self.norm = norm
        self.dropout_rate = dropout
        self.readout = readout
        self.random_sign = random_sign
        self.no_train = False

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            try:
                module.reset_parameters()
            except AttributeError:
                for x in module:
                    x.reset_parameters()

    def forward(self, x):
        if self.random_sign:
            sign_flip = torch.rand(x.size(0), 1, x.size(2)).to(x.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            sign_flip = torch.broadcast_to(sign_flip, x.size())
            x = x * sign_flip
        x = self.activation(self.lin1(x))
        if self.norm:
            x = self.layernorm(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        x = x.max(dim=1)[0] if self.readout == "max" else x.mean(dim=1)
        if self.no_train:
            x = x.detach()
        x = self.classifier(x)
        return x


class DeepSetsSignNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: bool, width1: int, width2: int, readout: str = "mean"):
        super(DeepSetsSignNet, self).__init__()
        self.in_channels = in_channels
        self.width2 = width2
        if norm:
            self.enc = torch.nn.Sequential(
                torch.nn.Linear(1, width1),
                torch.nn.LeakyReLU(),
                torch.nn.LayerNorm(width1, eps=1e-6),
                torch.nn.Linear(width1, width2),
                torch.nn.LeakyReLU()
            )
        else:
            self.enc = torch.nn.Sequential(
                torch.nn.Linear(1, width1),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(width1, width2),
                torch.nn.LeakyReLU()
            )
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(width2 * in_channels, in_channels),
            torch.nn.LeakyReLU()
        )
        self.classifier = DeepSets(in_channels, out_channels, norm, 0.0, width1, width2, torch.nn.LeakyReLU())
        self.norm = norm
        self.readout = readout
        self.no_train = False

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            try:
                module.reset_parameters()
            except AttributeError:
                for x in module:
                    x.reset_parameters()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.enc(x) + self.enc(-x)
        orig_shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.rho(x)
        x = x.reshape(orig_shape[0], orig_shape[1], self.in_channels)
        x = x.squeeze(-1)
        x = self.classifier(x)
        if self.no_train:
            x = x.detach()
        return x
