import torch
from torchsummary import summary
from kernel.models.denseNN import DenseNN



def viz_sin_phase_preds():
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
    model = DenseNN(11, 12, **config)
    model.load_model()

    return 