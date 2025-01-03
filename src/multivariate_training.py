import argparse
from hyperopt import hp, fmin, tpe, Trials

def define_space():

    space = {
    'batch_size': hp.choice('batch_size', [256]),
    'kernel_size': hp.choice('kernel_size',[[2,2,2],[4,4,4],[8,8,8],[8,4,1],[16,8,1]]),
    'expressiveness_ratio': hp.choice('expressiveness_ratio', [[1./168, 1./24, 1.], [1./24, 1./12, 1.], [1./180, 1./60, 1.],[1./40, 1./20, 1.],[1./64, 1./8, 1.]]),
    'nb_stack': hp.choice('nb_stack', [1]),  
    'nb_block': hp.choice('nb_block', [3]),  
    'input_size': hp.choice('input_size', [96*5]),     
    'horizon': hp.choice('horizon', [96]),           
    'hidden_sizes': hp.choice('hidden_sizes', [[512,512]]),  
    'learning_rate': hp.choice('learning_rate',[1e-3]),  
    'random_seed': hp.quniform('random_seed', 1, 10, 1)
    }

    return space 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset", required=False, default="Exchange", type=str, help="Name of the dataset")
    parser.add_argument("--horizon", required=False, default= 96, type=int, help="Horizon")
    args = parser.parse_args()
    pass

if __name__=='__main__':
    main()
    # La suite bientôt