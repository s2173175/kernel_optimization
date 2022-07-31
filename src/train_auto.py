import re
import torch

from kernel.models.denseNN import DenseNN

import sys
import argparse
import csv
import numpy as np

import optuna

def trainer(trial):
    config = {
        "learning_rate":trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "dropout_prob":trial.suggest_float("dropout_prob", 0.0, 0.3),
        "l2":trial.suggest_float("l2", 0.0, 0.1),
        "decay_rate":trial.suggest_categorical("decay_rate", [0.7,0.8,0.9]),
        "batch_size":trial.suggest_categorical("batch_size", [100,500,1000,2000]),
        "activation":trial.suggest_categorical("activation", ['tanh', 'relu']),
        "batch_norm":trial.suggest_categorical("batch_norm", [True, False]),
        "network_size":trial.suggest_categorical("network_size", ['small', 'medium', 'large', 'biggest']),
        "seed":500,
        "data_dir": ["./data/sets/train_2", "./data/sets/val_2"],
        "max_epoch":20,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "mode":"rollout-training",
    }
    
    networks = {
        'small':{'width':128,'depth':2 },
        'medium':{'width':128,'depth':3},
        'large':{'width':256,'depth':3},
        'biggest':{'width':256,'depth':4}
    }
    
    config['width'] = networks[config['network_size']]['width']
    config['depth'] = networks[config['network_size']]['depth']
    
    print(config)


    model = DenseNN(7, 12, **config)


    model.load_data()
    valid_best, losses, validations = model.train_model()
    
    return valid_best

if __name__ == "__main__":
    storage = "sqlite:///demo.db"
    
    try:
        study = optuna.create_study(study_name="hparams", storage=storage, sampler=optuna.samplers.TPESampler(), directions=['minimize'])
    except:
        study = optuna.load_study(study_name="hparams", storage=storage, sampler=optuna.samplers.TPESampler())
        
    study.optimize(trainer, n_trials=100, gc_after_trial=True, n_jobs=1)