# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:02:17 2024

@author: neman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class Block(nn.Module):
    def __init__(self, input_dim, kernel_size, r, hidden_units, H):
        """
        Parameters
        ----------
        input_dim : TYPE
            dimension du signal d'entrée.
        kernel_size : TYPE
            taille du kernel pour le maxpooling.
        expressiveness_ratio : TYPE
            ration pour définir la granularité de l'interpolation.
        hidden_units : TYPE
            nombre de neuronnes dans les couches cachées.
        horizon : TYPE
            nombre de pas de temps à prédire.

        """
        super(Block, self).__init__()
        self.kernel_size = kernel_size
        self.r = r
        self.H = H
        
        #Réseau MLP avec une couche cachée
        self.mlp = nn.Sequential(
            nn.Linear(input_dim // kernel_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, int(ceil(H * r)))
        )
        self.backcast_linear == nn.Linear(hidden_units, input_dim // kernel_size)
        
        
    def multirate_signal_sampling(self, y):
        """
        Parameters
        ----------
        y : TYPE
            Tensor du signal d'entrée (batch_size, input_dim).

        Returns
        -------
        y_subsampled.

        """
        y = y.unsqueeze(1)
        y_subsampled = F.max_poolid(y, kernel_size = self.kernel_size).squeeze(1)
        return y_subsampled
        
    def nonlinear_regression(self, y_subsampled):
        """
        Parameters
        ----------
        y_subsampled : TYPE
            Tensor du signal d'entrée avec max-pooling.

        Returns
        -------
        theta_f : TYPE
            coefficients pour le forecast.
        theta_b : TYPE
            coefficients pour le backcast.

        """
        hidden = self.mlp[:-1](y_subsampled)  # Projection non linéaire
        theta_f = self.mlp[-1](hidden)       # Projection pour le forecast
        theta_b = self.backcast_linear(hidden)  # Projection pour le backcast
        return theta_f, theta_b
    
    def hierarchical_interpolation(self, theta_f, theta_b):
        """
        

        Parameters
        ----------
        theta_f : TYPE
            coefficients pour le forecast.
        theta_b : TYPE
            coefficients pour le backcast.

        Returns
        -------
        y_forecast
        y_backcast

        """
        # Prévision via interpolation linéaire
        y_forecast = F.interpolate(
           theta_f.unsqueeze(1), size=self.H, mode='linear', align_corners=True
        ).squeeze(1)
       
        # Backcast via interpolation
        y_backcast = F.interpolate(
           theta_b.unsqueeze(1), size=self.kernel_size * theta_b.size(1), mode='linear', align_corners=True
        ).squeeze(1)
       
        return y_forecast, y_backcast
   
    def forward(self, y):
        y_subsampled = self.multirate_signal_sampling(y)
        theta_f, theta_b = self.nonlinear_regression(y_subsampled)
        y_forecast, y_backcast = self.hierarchical_interpolation(theta_f, theta_b)
        return y_forecast, y_backcast
