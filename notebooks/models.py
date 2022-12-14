import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( input_shape, hidden_units * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_units * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_units * 8, hidden_units * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( hidden_units * 4, hidden_units * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( hidden_units * 2, hidden_units, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(True),
            nn.ConvTranspose2d( hidden_units, output_shape, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_units, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_units, hidden_units * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_units * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_units * 2, hidden_units * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_units * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_units * 4, hidden_units * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_units * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_units * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)