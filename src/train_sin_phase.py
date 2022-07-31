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
        "data_dir": ["./data/sets/walking_cmd_v2_x.csv", "./data/sets/walking_cmd_v2_y.csv"],
        "batch_size":1000,
        "save_dir": "./kernel/results/walking_sin_phase",
        "log_file": "./kernel/results/walking_sin_phase/training_logs.out",
        "model_file": "./kernel/results/walking_sin_phase/checkpoint_best.pt",
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


    model = DenseNN(11, 12, **config)
    summary(model, input_data=(100,11),batch_dim=None)

    # model.load_data()
    # valid_best = model.train_model()
    #
    model.load_model()


    model.view_plot_foot_positions()
    # print(model.test_model())



    # if config['mode'] == "grid_search":
    #     with open('./kernel/grid_search/gs_walk_dest.csv', 'a') as f:
    #         writer = csv.writer(f)
        
    #         writer.writerow([config["learning_rate"], \
    #                 config["decay_rate"], \
    #                 config["depth"], \
    #                 config["width"], \
    #                 config["batch_size"], \
    #                 config["max_epoch"], \
    #                 config["seed"], \
    #                 valid_best
    #             ])


