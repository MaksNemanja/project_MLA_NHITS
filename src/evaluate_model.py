import os
import argparse
import pickle
import pandas as pd
import torch.nn as nn
import numpy as np
from hyperopt import hp, fmin, tpe, Trials,STATUS_OK
from torch.utils.data import DataLoader
from utils.objective_function import objective
from utils.preprocess import create_dataset
from utils.utils import getBestModelfromTrials, evaluate_model
from model.pytorch_model import NHITSModel

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False, default="Exchange", type=str, help="Name of the dataset")
    parser.add_argument("--horizon", required=False, default= 96, type=int, help="Horizon")
    args = parser.parse_args()

    return args

def define_space(args):

    space = {
    'batch_size': hp.choice('batch_size', [256]),
    'kernel_size': hp.choice('kernel_size',[[2,2,2],[4,4,4],[8,8,8],[8,4,1],[16,8,1]]),
    'expressiveness_ratio': hp.choice('expressiveness_ratio', [[1./168, 1./24, 1.], [1./24, 1./12, 1.], [1./180, 1./60, 1.],[1./40, 1./20, 1.],[1./64, 1./8, 1.]]),
    'nb_stack': hp.choice('nb_stack', [1]),  
    'nb_block': hp.choice('nb_block', [3]),  
    'input_size': hp.choice('input_size', [args.horizon*5]),     
    'horizon': hp.choice('horizon', [args.horizon]),           
    'hidden_sizes': hp.choice('hidden_sizes', [[512,512]]),  
    'learning_rate': hp.choice('learning_rate',[1e-3]),  
    'random_seed': hp.quniform('random_seed', 1, 10, 1)
    }

    return space 

def main(args):

    # ----------------- Import hyperopt Trial object from file ---------------------

    trial_file_path = f"../results/{args.dataset}/serie_{args.id}_{args.horizon}.pkl"
    
    if os.path.exists(trial_file_path):
        with open(trial_file_path, 'rb') as f:
            trial = pickle.load(f)
    else:
        print(f"Fichier {trial_file_path} n'existe pas.")
    
    # ----------------------- Trained model weights --------------------------------

    model_state_dict= trial["result"]["model_state_dict"]

    #------------------------ Paramètres du model ----------------------------------
    params= trial["result"]["params"]
    batch_size = params['batch_size']
    nb_stack = params['nb_stack']
    nb_block = params['nb_block']
    input_size = params['input_size']
    horizon = params['horizon']
    hidden_sizes = params['hidden_sizes']
    kernel_size = params['kernel_size']  
    expressiveness_ratio = params['expressiveness_ratio']  
    learning_rate = params['learning_rate']
    random_seed = params['random_seed']

    if args.dataset == 'ETTm2':
        val_ratio=0.20
    else:
        val_ratio=0.10
    
   # ------------------------ Create and Load Test Data -----------------------------
    _,_,test_dataset= create_dataset(univariate_data=args.univariate_data, 
                                                           input_size=5*horizon, 
                                                           horizon=horizon, 
                                                           train_ratio=0.7, 
                                                           val_ratio=val_ratio, 
                                                           Standardization=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # ------------------------ Instantiate new model and import weights and parameters from trained model -----------------------------
    model = NHITSModel(
        nb_stack, nb_block, input_size, horizon, hidden_sizes, kernel_size, expressiveness_ratio, learning_rate
    )
    model.load_state_dict(model_state_dict)

    mae,_,_ = evaluate_model(model, test_loader,loss_function= nn.L1Loss())
    mse,_,_ = evaluate_model(model, test_loader,loss_function= nn.MSELoss())
    print(f"------ Résultats pour ID={args.id} et Horizon={args.horizon} -------")
    print(f"MSE : {mse}")
    print(f"MAE : {mae}")
    
    return mse,mae


if __name__=='__main__':

    args= parse_args()
    path = f"../datasets/{args.dataset}/M/df_y.csv"
    df = pd.read_csv(path)
    multivariate_data= df["y"]
    unique_ids = df["unique_id"].unique()
    
    mse_list=[]
    mae_list=[]
    
    for id in unique_ids: # Itere sur les différentes séries du dataset
        args.univariate_data= multivariate_data[df['unique_id'] == id].values # serie temporelle univarié
        args.id= id
        mse,mae=main(args)
        mse_list.append(mse)
        mae_list.append(mae)
        #print(args.id)
    
    print(f"------ Multivariate Dataset Results -------")
    print(f"Average MSE : {np.mean(mse_list)}")
    print(f"Average MAE : {np.mean(mae_list)}")
    # La suite bientôt