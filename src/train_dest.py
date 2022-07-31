import re
import torch
from torchsummary import summary
from kernel.models.denseNN import DenseNN

import sys
import argparse
import csv

if __name__ == "__main__":

    config = {
        "learning_rate":1e-3,
        "dropout_prob":0.2,
        "l2":0,
        "max_epoch":20,
        "data_dir": ["./data/sets/walking_cmd_2_x.csv", "./data/sets/walking_cmd_2_y.csv"],
        "batch_size":1000,
        "save_dir": "./kernel/results/walking_cmd_com_2_test",
        "log_file": "./kernel/results/walking_cmd_com_2_test/training_logs.out",
        "model_file": "./kernel/results/walking_cmd_com_2_test/checkpoint_best.pt",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "mode":"training",
        "seed":0,
        "decay_rate":0
    }




    parser = argparse.ArgumentParser()



    parser.add_argument('-lr', help='foo help', default=0.001, type=float)
    parser.add_argument('-decay', help='foo help',default=0.8, type=float)
    parser.add_argument('-depth', help='foo help',default=3, type=int)
    parser.add_argument('-width', help='foo help',default=256, type=int)
    parser.add_argument('-batch_size', help='foo help',default=100, type=int)
    parser.add_argument('-seed', help='foo help', default=None, type=int)
    parser.add_argument('-epochs', help='foo help',default=20, type=int)
    parser.add_argument('-mode', help='foo help',default='rollout-training', type=str)

    args = parser.parse_args()


    config["learning_rate"] = args.lr
    config["decay_rate"] = args.decay
    config["depth"] = args.depth
    config["width"] = args.width
    config["batch_size"] = args.batch_size
    config["max_epoch"] = args.epochs
    config["seed"] = args.seed
    config["mode"] = args.mode



    print(config)


    model = DenseNN(7, 12, **config)
    summary(model, input_data=(100,7),batch_dim=None)

    model.load_data()
    valid_best = model.train_model()
 

