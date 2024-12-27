# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 09:04:08 2024

@author: neman
"""
#pour ECL prendre les 168 premiers valeurs

import csv
import pandas as pd
import torch
from torch.utils.data import TensorDataset

def load_data(dataset):
    # Ouvrir le fichier CSV
    with open(dataset, newline='') as csvfile:
        # Créer un lecteur CSV
        csvreader = csv.reader(csvfile, delimiter=',')
        data = list(csvreader)
        data = data[1::]
        for i, line in enumerate(data):
            new_line = []
            for x in line:
                try:
                    new_line.append(float(x.strip()))
                except ValueError:
                    None
            data[i] = new_line

            """
            mini = min(data[i])
            maxi = max(data[i])
            for k, j in enumerate(data[i]):
                j = (j - mini) / (maxi - mini)
                data[i][k] = j
            """
            
    #x_data = torch.tensor([row[:-68] for row in data], dtype=torch.float32)  # Toutes les colonnes sauf la dernière
    #y_data = torch.tensor([row[-68:] for row in data], dtype=torch.float32)   # Dernière colonne (cible)
    x_data = torch.tensor(data, dtype = torch.float32)
    return x_data
    #return TensorDataset(x_data, y_data)
            


