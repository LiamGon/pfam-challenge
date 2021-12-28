import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.module(x) + self.beta * x


class Resnet(nn.Module):
    def __init__(self, input_dim, F, kernel_size, n_layers, n_classes):

        super().__init__()
        self.F = F
        self.kernel_size = kernel_size
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.input_dim = input_dim
        self.embedding = nn.Sequential(
            nn.Conv1d(
                input_dim,
                F,
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm1d(self.F),
        )

        self.conv_layers = nn.Sequential(
            *[SkipLayer(self.get_conv_layer()) for k in range(n_layers)]
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.F), nn.Linear(F, n_classes)
        )

    def get_conv_layer(self):
        return nn.Sequential(
            nn.Conv1d(
                self.F,
                self.F,
                kernel_size=self.kernel_size,
                padding=self.kernel_size-1,
                dilation=2
            ),
            self.activation,
            nn.BatchNorm1d(self.F),
        )
    def get_hidden_rep(self, seqs):
        seqs = (
        F.one_hot(seqs, num_classes=self.input_dim)
        .permute(0, 2, 1)
        .type(torch.float)
    )
        h = self.embedding(seqs)

        h = self.conv_layers(h)

        h = h.mean(dim=2)
        
        return h
    
    def forward(self, seqs):

        h = self.get_hidden_rep(seqs)
        
        pred = self.classifier(h)

        return pred