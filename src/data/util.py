from copyreg import pickle
from matplotlib import container
import numpy as np
import pandas as pd
import random


def split_data():
    
    x = pd.read_csv('./sets/walking_cmd_2_x.csv', index_col=0).to_numpy()
    y = pd.read_csv('./sets/walking_cmd_2_y.csv', index_col=0).to_numpy()
    
    data = list(zip(x,y))
    random.shuffle(data)
    
    N = len(data)
    
    training_data = data[:int(0.8*N)]
    validation_data = data[int(0.8*N):]
    
    train_x, train_y = zip(*training_data)
    val_x, val_y = zip(*validation_data)
    
    train_x = pd.DataFrame(np.array(train_x))
    train_y = pd.DataFrame(np.array(train_y))
    
    val_x = pd.DataFrame(np.array(val_x))
    val_y = pd.DataFrame(np.array(val_y))
    
    train_x.to_csv('./sets/train_2_x.csv')
    train_y.to_csv('./sets/train_2_y.csv')
    val_x.to_csv('./sets/val_2_x.csv')
    val_y.to_csv('./sets/val_2_y.csv')
    

    return


split_data()