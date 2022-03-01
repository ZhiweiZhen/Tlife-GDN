# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import time
from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep
import random
from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random



class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 

        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)

      
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())
        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
            'pred_ahead': train_config['pred_ahead'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        dataset_size = len(test_dataset)
        dataset_indices = list(range(train_config['slide_win']-train_config['days_ahead'],dataset_size+train_config['slide_win']-train_config['days_ahead']))
        #dataset_indices = list(range(dataset_size))
        
        train_dataloader, val_dataloader, train_sampler, val_sampler = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])
        folderSaveImgs = os.path.join('/content/drive/MyDrive/GDN/data/', dataset,'PI_TRAIN/')
        train_PI = np.load(folderSaveImgs+'ATD_PIs.npy',allow_pickle = True)
        train_PIloader = DataLoader(dataset=train_PI, shuffle=False, batch_size=train_config['batch'], sampler=train_sampler)
        
        val_PIloader = DataLoader(dataset=train_PI, shuffle=False, batch_size=train_config['batch'], sampler=val_sampler)
        folderSaveImgs = os.path.join('/content/drive/MyDrive/GDN/data/', dataset,'PI_TEST/')

        test_PI = np.load(folderSaveImgs+'ATD_PIs.npy',allow_pickle = True)
        test_PI = torch.utils.data.Subset( test_PI, dataset_indices)
        test_PIloader = DataLoader(dataset=test_PI, shuffle=False, batch_size=train_config['batch'])
        
        
        
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False)
                            
        

        self.train_PIdataloader = train_PIloader
        self.val_PIdataloader = val_PIloader
        self.test_PIdataloader = test_PIloader

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)

    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            
            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset'],
                train_PIloader=self.train_PIdataloader,
                test_PIloader=self.test_PIdataloader,
                val_PIloader=self.val_PIdataloader
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader,self.test_PIdataloader)
        _, self.val_result = test(best_model, self.val_dataloader,self.val_PIdataloader)
   
        info = self.get_score(self.test_result, self.val_result)
        
        auc_save_path = self.get_save_path()[1]
        np.savetxt(auc_save_path,info[5])
        
        error_save_path = self.get_save_path()[2]
        np.savetxt(error_save_path,info[6])

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):

        dataset_size = len(train_dataset)

        dataset_indices = list(range(dataset_size))
        np.random.shuffle(dataset_indices)
        val_split_index = int(np.floor(val_ratio * dataset_size))
        train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx) 

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx) 
        train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=train_config['batch'], sampler=train_sampler)
        val_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=train_config['batch'], sampler=val_sampler)

        return train_loader, val_loader,train_sampler,val_sampler
    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)
        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        #print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')
        return info

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./error_socre/{dir_path}/{datestr}.csv',
            f'./auc_score/{dir_path}/{datestr}.txt'
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=30)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-pred_ahead', help='pred_ahead', type = int, default=3)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=10)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='msl')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=random.randint(0, 99))
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-lr', help='lr', type = float, default=0.000001)
    parser.add_argument('-days_ahead', help='days ahead for persistent image', type = int, default=0)

    args = parser.parse_args()
    print(args.pred_ahead)
    print(args.days_ahead)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'pred_ahead': args.pred_ahead,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'lr': args.lr,
        'days_ahead': args.days_ahead,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()





