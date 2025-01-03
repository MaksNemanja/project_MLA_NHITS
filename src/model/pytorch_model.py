import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from model.nhits import NHITS


class NHITSModel(pl.LightningModule):
    def __init__(self, nb_stack, nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio, learning_rate=1e-3):
        super(NHITSModel, self).__init__()
        self.save_hyperparameters()
        self.model = NHITS(nb_stack, nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio)
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        #print(f"Training loss: {loss}")
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch  
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        #print(f"Validation loss: {loss.item()}")
        self.log("Validation_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):

        optimizer= optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_decay_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=333, gamma=0.5)
        return [optimizer],[lr_decay_scheduler]