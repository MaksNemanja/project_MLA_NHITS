# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:14:46 2024

@author: neman
"""

import torch
import torch.nn as nn
from stack import Stack

class NHiTSModel(nn.Module):
    def __init__(self, num_stacks, num_blocks_per_stack, input_dim, kernel_size, 
                 expressiveness_ratios, hidden_units, H):
        """
        Parameters
        ----------
        - num_stacks : int
            Nombre de piles dans le modèle.
        - num_blocks_per_stack : int
            Nombre de blocs dans chaque pile.
        - input_dim : TYPE
            Dimension de l'entrée (longueur des séries temporelles).
        - kernel_size : TYPE
            Taille du kernel pour le max-pooling multi-taux.
        - expressiveness_ratios : TYPE
            Liste des ratios d'expressivité.
        - hidden_units : TYPE
            Nombre de neurones dans les couches cachées des blocs.
        - H : TYPE
            Nombre de pas de temps à prédire.
        """
        super(NHiTSModel, self).__init__()
        

        # Initialise les piles (stacks)
        self.stacks = nn.ModuleList([
            Stack(num_blocks=num_blocks_per_stack, 
                  input_dim=input_dim,
                  kernel_size=kernel_size, 
                  r=expressiveness_ratios[i], 
                  hidden_units=hidden_units, 
                  H=H) 
            for i in range(num_stacks)
        ])
        self.num_stacks = num_stacks

    def forward(self, y, return_stack_forecast=None):
        """
        Parameters
        ----------
        - y : TYPE
            Signal d'entrée (batch_size, input_dim).
        - return_stack_forecast : int
            Indice de la pile dont on veut récupérer la prévision.

        Return
        ------
        - global_forecast : TYPE
            Prévision combinée de toutes les piles (batch_size, horizon).
        - stack_forecast : TYPE
            Prévision d'une pile spécifique si return_stack_forecast est défini.
        """
        
        stack_forecasts = []
        stack_residual = y.clone() 
        
        for stack in self.stacks:
            stack_forecast, stack_residual = stack(stack_residual)
            stack_forecasts.append(stack_forecast)

        # Prévision globale
        global_forecast = torch.sum(torch.stack(stack_forecasts), dim=0)
        
        if return_stack_forecast is not None:
            # Retourne uniquement la prévision d'une pile spécifique
            return global_forecast, stack_forecasts[return_stack_forecast]

        return global_forecast
