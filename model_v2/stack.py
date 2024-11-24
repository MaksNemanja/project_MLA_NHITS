# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:50:47 2024

@author: neman
"""

import torch
import torch.nn as nn
from block import Block

class Stack(nn.Module):
    def __init__(self, num_blocks, input_dim, kernel_size, r, hidden_units, H):
        """
        Parameters
        ----------
        - num_blocks : TYPE
            Nombre de blocs dans la pile.
        - input_dim : TYPE
            Dimension du signal d'entrée.
        - kernel_size : TYPE
            Taille du kernel pour le max-pooling multi-taux.
        - r : TYPE
            Ratio définissant la granularité d'interpolation dans chaque bloc.
        - hidden_units : TYPE
            Nombre de neurones dans les couches cachées de chaque bloc.
        - H : TYPE
            Nombre de pas de temps à prédire.
        """
        super(Stack, self).__init__()
        self.blocks = nn.ModuleList([
            Block(input_dim, kernel_size, r, hidden_units, H) 
            for _ in range(num_blocks)
        ])
        self.num_blocks = num_blocks

    def forward(self, y):
        """
        Parameters
        ----------
        - y : TYPE
            Signal d'entrée (batch_size, input_dim).
        
        Returns 
        -------
        - stack_forecast : TYPE
            Prévisions cumulées de tous les blocs (batch_size, horizon).
        - stack_residual : TYPE
            Résidu final du signal, à passer aux prochains stacks (batch_size, input_dim).
        """
        stack_forecast = torch.zeros(y.size(0), self.blocks[0].H, device=y.device)  # Initialisation
        stack_residual = y.clone()  # Initialisation du résidu comme le signal d'entrée
        
        for block in self.blocks:
            block_forecast, block_backcast = block(stack_residual)  # Passage du résidu à chaque bloc
            
            # Mise à jour du résidu (y_t+1 = y_t - backcast)
            stack_residual = stack_residual - block_backcast
            
            # Accumulation des prévisions des blocs dans la pile
            stack_forecast = stack_forecast + block_forecast

        return stack_forecast, stack_residual
