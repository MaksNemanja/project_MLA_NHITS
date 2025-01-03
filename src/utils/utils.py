import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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
    print(f"Validation MAE (Mean Absolute Error): {mean_loss}")

    return mean_loss, np.concatenate(all_y_true), np.concatenate(all_y_pred)

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