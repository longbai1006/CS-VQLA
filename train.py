import os
import argparse
import json
import torch
import torch.utils.data
from torch.utils.data  import DataLoader
from utils.vqla import *
from dataloader.dataloader import EndoVis18VQLA, EndoVis17VQLA, M2CAIVQLA
from models.continual import CSVQLA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/csvqla.json',
                        help='Json file of settings.')

    return parser

if __name__ == '__main__':
    
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  
    args.update(param)  # Add parameters from json
    _set_device(args)

    batch_size = args["batch_size"]
    folder_tail = '/vqla/Classification/*.txt'

    # data location
    train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    val_seq = [1, 5, 16]
    endo18_folder_head = 'data/EndoVis-18-VQLA/seq_'
    # dataloader
    train_dataset_step0 = EndoVis18VQLA(train_seq, endo18_folder_head, folder_tail, patch_size = 5)
    train_loader_step0 = DataLoader(dataset=train_dataset_step0, batch_size=batch_size, shuffle=True)
    test_dataset_step0 = EndoVis18VQLA(val_seq, endo18_folder_head, folder_tail, patch_size = 5)
    test_loader_step0 = DataLoader(dataset=test_dataset_step0, batch_size=batch_size, shuffle=False)
    
    mi2cai_train_folder_head = 'data/m2cai2016-VQLA/train/'
    mi2cai_test_folder_head = 'data/m2cai2016-VQLA/test/'
    train_dataset_mi2cai = M2CAIVQLA(mi2cai_train_folder_head, folder_tail, patch_size=5)
    train_loader_step2 = DataLoader(dataset=train_dataset_mi2cai, batch_size=batch_size, shuffle=False)
    test_dataset_mi2cai = M2CAIVQLA(mi2cai_test_folder_head, folder_tail, patch_size=5)
    test_loader_step2 = DataLoader(dataset=test_dataset_mi2cai, batch_size=batch_size, shuffle=False)

    endo17_train_folder_head = 'data/EndoVis-17-VQLA/train/'
    endo17_test_folder_head = 'data/EndoVis-17-VQLA/test/'
    train_dataset_endo17 = M2CAIVQLA(endo17_train_folder_head, folder_tail, patch_size = 5)
    train_loader_step1 = DataLoader(dataset=train_dataset_endo17, batch_size=batch_size, shuffle=False)
    test_dataset_endo17 = M2CAIVQLA(endo17_test_folder_head, folder_tail, patch_size = 5)
    test_loader_step1 = DataLoader(dataset=test_dataset_endo17, batch_size=batch_size, shuffle=False)

    model = CSVQLA(args)

    for task_id in 0,1,2:
        model.incremental_train(task_id, 
                            train_loader_step0, 
                            train_loader_step1,
                            train_loader_step2,
                            test_loader_step0,
                            test_loader_step1,
                            test_loader_step2)
        model.after_task()
