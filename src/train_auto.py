import re
import torch
from torchsummary import summary
from kernel.models.denseNN import DenseNN

import sys
import argparse
import csv
import numpy as np

if __name__ == "__main__":

    config = {
        "learning_rate":1e-3,
        "dropout_prob":0.2,
        "l2":0,
        "max_epoch":20,
        "data_dir": ["./data/sets/walking_cmd_100_x.csv", "./data/sets/walking_cmd_100_y.csv"],
        "batch_size":1000,
        "save_dir": "./kernel/results/walking_cmd_100",
        "log_file": "./kernel/results/walking_cmd_100/training_logs.out",
        "model_file": "./kernel/results/walking_cmd_100/checkpoint_best.pt",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "mode":"training",
        "seed":0,
        "decay_rate":0
    }




    parser = argparse.ArgumentParser()
    parser.add_argument('-name', help='foo help',required=True, type=str)
    parser.add_argument('-input_dims', help='foo help',required=True, type=int)



    parser.add_argument('-lr', help='foo help', default=0.001, type=float)
    parser.add_argument('-decay', help='foo help',default=0.8, type=float)
    parser.add_argument('-depth', help='foo help',default=3, type=int)
    parser.add_argument('-width', help='foo help',default=256, type=int)
    parser.add_argument('-batch_size', help='foo help',default=100, type=int)
    parser.add_argument('-seed', help='foo help', default=None, type=int)
    parser.add_argument('-epochs', help='foo help',default=20, type=int)
    parser.add_argument('-mode', help='foo help',default='rollout-training', type=str)

    args = parser.parse_args()



    x = f"./data/sets/{args.name}_x.csv"
    y = f"./data/sets/{args.name}_y.csv"
    config['data_dir'] = [x,y]

    save = f"./kernel/results/{args.name}"
    log =  f"./kernel/results/{args.name}/training_logs.out"
    model =  f"./kernel/results/{args.name}/checkpoint_best.pt"

    config["save_dir"] = save
    config["log_file"] = log
    config["model_file"] = model

    config["learning_rate"] = args.lr
    config["decay_rate"] = args.decay
    config["depth"] = args.depth
    config["width"] = args.width
    config["batch_size"] = args.batch_size
    config["max_epoch"] = args.epochs
    config["seed"] = args.seed
    config["mode"] = args.mode



    print(config)


    model = DenseNN(args.input_dims, 12, **config)
    summary(model, input_data=(100,args.input_dims),batch_dim=None)

    model.load_data()
    valid_best, losses, validations = model.train_model()
 

    with open("./kernel/results/losses_validations_all.csv", 'a') as f:
        writer = csv.writer(f)

        for step in range(len(losses)):

            # row = np.concatenate(args.name, step, losses[step],validations[step])
            writer.writerow([args.name, step, losses[step],validations[step]])