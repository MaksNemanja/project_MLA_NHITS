import os
import argparse
import pickle
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials,STATUS_OK
from utils.objective_function import objective
from utils.preprocess import create_dataset
from utils.utils import getBestModelfromTrials

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False, default="ECL", type=str, help="Name of the dataset")
    parser.add_argument("--horizon", required=False, default= 96, type=int, help="Horizon")
    args = parser.parse_args()

    return args

def define_space(args):

    space = {
    'batch_size': hp.choice('batch_size', [256]),
    'kernel_size': hp.choice('kernel_size',[[2,2,2],[4,4,4],[8,8,8],[8,4,1],[16,8,1]]),
    'expressiveness_ratio': hp.choice('expressiveness_ratio', [[1./168, 1./24, 1.], [1./24, 1./12, 1.], [1./180, 1./60, 1.],[1./40, 1./20, 1.],[1./64, 1./8, 1.]]),
    'nb_stack': hp.choice('nb_stack', [3]),  
    'nb_block': hp.choice('nb_block', [1]),  
    'input_size': hp.choice('input_size', [args.horizon*5]),     
    'horizon': hp.choice('horizon', [args.horizon]),           
    'hidden_sizes': hp.choice('hidden_sizes', [[512,512]]),  
    'learning_rate': hp.choice('learning_rate',[1e-3]),
    'shuffle': hp.choice('shuffle',[True,False]),  
    'random_seed': hp.quniform('random_seed', 1, 10, 1)
    }

    return space 

def main(args):


    space= define_space(args)
    
    if args.dataset == 'ETTm2':
        val_ratio=0.20
    else:
        val_ratio=0.10
    
    train_dataset,val_dataset,_= create_dataset(univariate_data=args.univariate_data, 
                                                           input_size=5*args.horizon, 
                                                           horizon=args.horizon, 
                                                           train_ratio=0.7, 
                                                           val_ratio=val_ratio, 
                                                           Standardization=True)

    trials = Trials()

    # Lance l'optimisation
    best = fmin(
        fn=lambda x: objective(x, train_dataset,val_dataset),  
        space=space,    # Espace de recherche
        algo=tpe.suggest,  # Algorithme 
        max_evals=10,   # Nombre de runs
        trials=trials    
        )
    
    best_trial=getBestModelfromTrials(trials)
    # Affichage de la meilleur loss 
    print(f"BEST Validation MAE for the Temporal Series {args.id}:", best_trial["result"]["loss"])
    
    output_dir= f"../results/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/serie_{args.id}_{args.horizon}.pkl"

    # On sauvegarde l'objet trial contenant les poids du meilleur model et ses paramètres
    with open(file_path, 'wb') as f:
        pickle.dump(best_trial, f) 
    
    

if __name__=='__main__':

    args= parse_args()
    path = f"../datasets/{args.dataset}/M/df_y.csv"
    df = pd.read_csv(path)
    multivariate_data= df["y"]
    unique_ids = df["unique_id"].unique()
    

    for id in unique_ids: # Itere sur les différentes séries du dataset
        args.univariate_data= multivariate_data[df['unique_id'] == id].values # serie temporelle univarié
        args.id= id
        if args.dataset=="weather":
            args.id= unique_ids.index(id) # 
        main(args)
        #print(args.id)
        
    