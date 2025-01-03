
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def split_dataset(univariate_data,train_ratio=0.7,val_ratio=0.10,Standardization=True):

    total_size= len(univariate_data)
    train_end=int(total_size*train_ratio)
    val_end= int(total_size*(train_ratio + val_ratio))

    # Splitting des données 
    train_data= univariate_data[0:train_end]
    val_data= univariate_data[train_end:val_end]
    test_data= univariate_data[val_end:]

    # Normalisation des set d'entrainement
    if Standardization:

        # Calcul de la moyenne et écart type sur les données d'entrainement pour la normalisation
        mean = np.mean(train_data)
        std = np.std(train_data)
        
        train_data_normalized = (train_data - mean) / std
        val_data_normalized = (val_data - mean) / std
        test_data_normalized = (test_data - mean) /std

        return train_data_normalized,val_data_normalized,test_data_normalized
    else: return train_data,val_data,test_data

def create_sequences(data, input_size, horizon):
    """
    Générez des séquences pour correspondre aux dimensions d'entrée et de sortie du modèle.

    Args:
        data : serie temporelle 
        input_size (int): Taille de la fenêtre d'entrée.
        horizon (int): Taille de l'horizon de prévision.

    Returns:
        torch.Tensor, torch.Tensor: Séquences d'entrée (x) et de sortie (y).
    """
    x, y = [], []
    for i in range(len(data) - input_size - horizon):
        x.append(data[i : i + input_size])  # Fenêtre d'entrée
        y.append(data[i + input_size : i + input_size + horizon])  # Fenêtre de sortie (horizon)

    return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

def create_dataset(univariate_data,input_size, horizon, train_ratio=0.7, val_ratio=0.10, Standardization=True):

    train_data,val_data,test_data= split_dataset(univariate_data,train_ratio,val_ratio,Standardization)
    X_train,Y_train= create_sequences(train_data, input_size, horizon)
    X_val,Y_val= create_sequences(val_data, input_size, horizon)
    X_test,Y_test= create_sequences(test_data, input_size, horizon)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    return train_dataset,val_dataset,test_dataset