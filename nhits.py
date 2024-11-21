import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from math import ceil


class Block(nn.Module):
    def __init__(self, input_size,horizon, hidden_sizes, kernel_size, expressiveness_ratio):
        super(Block, self).__init__()

        self.L= input_size
        self.H= horizon
        self.k_l= kernel_size
        self.r_l= expressiveness_ratio
        self.theta_size= int(ceil(self.r_l * self.H))
        
        # MaxPool layer for multi rate signal sampling
        self.maxpool = nn.MaxPool1d(kernel_size=self.k_l)
        
        self.layers = nn.ModuleList() # To handle a list of layers

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        
        self.linear_f = nn.Linear(hidden_sizes[-1], self.theta_size)
        self.linear_b = nn.Linear(hidden_sizes[-1], self.theta_size)

    def forward(self, x):

        x_pooled = self.maxpool(x) 

        h= x_pooled
        for layer in self.layers:  
            h = layer(x_pooled)

        theta_f = self.linear_f(h).unsqueeze(1) # On unsqueeze car l'input d'interpolate doit être de la forme [batch, channel, data]
        theta_b = self.linear_b(h).unsqueeze(1)

        forecast= F.interpolate(theta_f, size= self.H, mode="linear").squeeze(1) # squeeze pour repasser de [batch, channel, data] à -> [batch, data]
        backcast= F.interpolate(theta_b, size=self.L, mode="linear").squeeze(1)
        
        return forecast, backcast