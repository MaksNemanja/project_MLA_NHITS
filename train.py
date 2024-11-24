from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from nhits import Block, Stack, NHITS

batch_size = 4
input_size = 100  # Taille de l'entrée
horizon = 50      # Taille de la prévision
hidden_sizes = [128, 128]
kernel_size = 2
expressiveness_ratio = 0.5
nb_block = 2
nb_stack = 3


x_data = torch.rand(100, input_size)  # 100 échantillons, taille entrée = input_size
y_data = torch.rand(100, horizon)    # 100 échantillons, taille sortie = horizon

# Création des DataLoaders
dataset = TensorDataset(x_data, y_data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        print(f"Training loss: {loss.item()}")
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    

model = NHITSModel(
    nb_stack, nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio
)

trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1)

# Entrainement
trainer.fit(model, train_loader)