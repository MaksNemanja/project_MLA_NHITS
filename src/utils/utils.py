import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK


def evaluate_model(model, dataloader,loss_function= nn.L1Loss() ):
    model.eval()  # Basculer en mode évaluation
    criterion = loss_function  # MAE
    total_loss = 0.0
    num_batches = 0

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():  # Pas de gradient pendant l'évaluation
        for batch in dataloader:
            x, y = batch
            y_pred = model(x)  # Prédiction
            loss = criterion(y_pred, y)  # Calcul de la perte
            total_loss += loss.item()
            num_batches += 1

            all_y_true.append(y.cpu().numpy())  # Stocker les vraies valeurs
            all_y_pred.append(y_pred.cpu().numpy())  # Stocker les prédictions

    # Moyenne des pertes
    mean_loss = total_loss / num_batches
    #print(f"Mean Loss sur le set de validation entier: {mean_loss}")

    return mean_loss, np.concatenate(all_y_true), np.concatenate(all_y_pred)


def getBestModelfromTrials(trials):
    # Filtrer les essais valides
    valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
    
    # Vérifier si des essais valides existent
    if not valid_trial_list:
        raise ValueError("Aucun essai valide trouvé.")

    # Extraire les pertes des essais valides
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    
    # Trouver l'indice de l'essai avec la perte minimale
    index_having_minumum_loss = np.argmin(losses)
    
    # Récupérer l'objet de l'essai ayant la perte minimale
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    
    return best_trial_obj

def plot_predictions(input_seq, predictions,true_seq):
        plt.figure()

        # Longueur totale : input + prediction
        total_length = len(input_seq) + len(predictions)

        # Tracé de l'entrée
        plt.plot(range(len(input_seq)), input_seq, label="Entrée (Input)", color="blue")

        # Tracé des prédictions (à partir de la fin de l'entrée)
        plt.plot(range(len(input_seq), total_length), predictions, label="Prédictions (Predictions)", color="orange")
        plt.plot(range(len(input_seq), total_length), true_seq, label="Vrai séquence", color="red")

        plt.legend()
        plt.show()

