import torch.optim as optim
from dslh import *


def get_config():
    config = {
        "start_epoch": 0,
        # "eta_threshold": 1.0,
        "eta_threshold": 0.7,
        # "eta_threshold": 0.55,
        "density_threshold": 8,
        "s2_scale": 0.4,
        # "s2_scale": 0.1,
        "alpha": 0.1,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[standard]",
        "step_continuation": 10,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar-10",
        "dataset": "nuswide_21",
        # "dataset":"coco",
        # "dataset":"nuswide_81",
        # "dataset":"imagenet",

        "epoch": 100,
        "evaluate_freq": 10,
        "GPU": True,
        # "GPU":False,
        "bit_list": [64, 48, 32, 16],
    }
    if config["dataset"] == "cifar-10":
        config["topK"] = 54000
        config["n_class"] = 10
    elif config["dataset"] == "nuswide_21":
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    config["data_path"] = "../b/data/" + config["dataset"] + "/"
    if config["dataset"][:7] == "nuswide":
        config["data_path"] = "../b/data/nuswide_81/"
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    for i in range(config["n_class"]):
        data_set = "class" + str(i + 1)
        config["data"][data_set] = {"list_path": "./data/" + config["dataset"] + "/" + data_set + ".txt",
                                    "batch_size": config["batch_size"]}
    return config
