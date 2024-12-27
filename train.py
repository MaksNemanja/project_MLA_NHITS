from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from nhits import Block, Stack, NHITS
#from NHITS_alg import *
from lecture_csv import load_data
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import MeanSquaredError, MeanAbsoluteError

batch_size = 4
input_size = 4  # Taille de l'entrée
horizon = 96      # Taille de la prévision
hidden_sizes = [128, 128]
kernel_size = 2
expressiveness_ratio = 0.5
nb_block = 2
nb_stack = 3


#x_data = torch.rand(100, input_size)  # 100 échantillons, taille entrée = input_size
#y_data = torch.rand(100, horizon)    # 100 échantillons, taille sortie = horizon

# Création des DataLoaders
#dataset = TensorDataset(x_data, y_data)
x_data = load_data('data/ETTm2/df_x.csv')
y_data = load_data('data/ETTm2/df_y.csv')

def create_sequences(x_data, y_data, input_size, horizon):
    """
    Générez des séquences pour correspondre aux dimensions d'entrée et de sortie du modèle.

    Args:
        x_data (torch.Tensor): Données des entrées (features), de type torch.Tensor.
        y_data (torch.Tensor): Données des cibles (valeurs à prédire), de type torch.Tensor.
        input_size (int): Taille de la fenêtre d'entrée (nombre de pas de temps).
        horizon (int): Taille de l'horizon de prévision (nombre de pas de temps).

    Returns:
        torch.Tensor, torch.Tensor: Séquences d'entrée (x) et de sortie (y).
    """
    x, y = [], []
    for i in range(len(x_data) - input_size - horizon + 1):
        x.append(x_data[i : i + input_size])  # Fenêtre d'entrée
        y.append(y_data[i + input_size : i + input_size + horizon])  # Fenêtre de sortie (horizon)

    return torch.stack(x), torch.stack(y)


# Charger les données (df_x pour les entrées, df_y pour les cibles)
x_data, y_data = create_sequences(x_data, y_data, input_size=input_size, horizon=horizon)

x_data = x_data.mean(dim=2)
        
y_data = y_data.squeeze(-1)

print("x size :",x_data.size())
print("y size :",y_data.size())
# Division des données
train_x, val_x, train_y, val_y = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Datasets et DataLoaders
train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for x,y in train_loader:
    print(x.size())
    print(y.size())
    break


class NHITSModel(pl.LightningModule):
    def __init__(self, nb_stack, nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio, learning_rate=1e-3):
        super(NHITSModel, self).__init__()
        self.save_hyperparameters()
        self.model = NHITS(nb_stack, nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        
        # Ajout des métriques
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        x = x.view(batch_size, -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        
        loss = self.criterion(y_pred, y)
        # Calcul des métriques pour l'entraînement
        self.train_mse.update(y_pred, y)
        self.train_mae.update(y_pred, y)
        
        print(f"Training loss: {loss.item()}")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_mse', self.train_mse, prog_bar=True)
        self.log('train_mae', self.train_mae, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        # Calcul des métriques pour la validation
        self.val_mse.update(y_pred, y)
        self.val_mae.update(y_pred, y)

        # Logging
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mse', self.val_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mae', self.val_mae, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def on_validation_epoch_end(self):
        # À la fin de chaque époque, calculez les métriques
        mse = self.val_mse.compute()
        mae = self.val_mae.compute()

        print(f"Validation MSE: {mse}")
        print(f"Validation MAE: {mae}")

        # Réinitialisez les métriques pour la prochaine époque
        self.val_mse.reset()
        self.val_mae.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

 
    
    
model = NHITSModel(
        nb_stack, nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio
    )
    

logger = CSVLogger(save_dir="logs", name="nhits_logs")
trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1)
 
torch.autograd.set_detect_anomaly(True)   
# Entrainement
trainer.fit(model, train_loader, val_loader)
#trainer.test(model, dataloaders=val_loader)

print(f"Test MSE: {model.val_mse.compute().item()}")
print(f"Test MAE: {model.val_mae.compute().item()}")
