import pytorch_lightning as pl
import torch.nn as nn
from model.pytorch_model import NHITSModel
from utils.utils import evaluate_model
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping

def objective(params, train_dataset,val_dataset,test_dataset):
    # Extraire les paramètres choisis par Hyperopt

    batch_size = params['batch_size']
    nb_stack = params['nb_stack']
    nb_block = params['nb_block']
    input_size = params['input_size']
    horizon = params['horizon']
    hidden_sizes = params['hidden_sizes']
    kernel_size = params['kernel_size']  # Directement une liste ou un tuple
    expressiveness_ratio = params['expressiveness_ratio']  # Directement un tuple
    learning_rate = params['learning_rate']
    random_seed = params['random_seed']
    
    #Set la seed avant un eventuel shuffle du dataloader :
    pl.seed_everything(random_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Initialiser le modèle NHITS avec ces paramètres
    model = NHITSModel(
        nb_stack, nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio, learning_rate
    )

    # Initialiser le trainer de PyTorch Lightning
    early_stop_callback = EarlyStopping(monitor="Validation_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    trainer = pl.Trainer(max_steps=1000,callbacks=[early_stop_callback], log_every_n_steps=1, val_check_interval=1.0)
    # Entraîner le modèle
    trainer.fit(model, train_loader, val_loader)

    # Extraire la perte de validation
    # Vous pouvez utiliser `trainer.callback_metrics` pour obtenir les métriques après l'entraînement
    val_loss,_,_ = evaluate_model(model,val_loader)
    test_loss,_,_ = evaluate_model(model,test_loader)
    test_loss,_,_ = evaluate_model(model,test_loader,nn.MSELoss())

    # Retourner la perte (c'est ce que Hyperopt va minimiser)
    return val_loss