import os
import torch
import pandas as pd
from utils import ror_ebgm
from embedding import Edge_Regression

# Preprocessing: alert signal for ROR|EBGM

args_preprocessing = {'original_data_path': './Data_ref/',
                      'out_data_path': './Data/',
                      'label_folder': 'split_year',
                      'graph_type': 'Directed',
                      'generate_core': True
                      }
def preprocessing(args):
    for cato in os.listdir(args['original_data_path']):

        print('Preprocessing for ' + cato)

        theta_init = pd.DataFrame({'alpha1': [0.2, 0.1, 0.1, 0.01, 0.01], 'beta1': [0.1, 0.1, 0.2, 0.2, 0.1], 'alpha2': [2, 10, 2, 8, 4],
                                   'beta2': [4, 10, 4, 10, 4], 'p': [1 / 3, 0.2, 1 / 3, 0.2, 0.2]})
        data_ref = args['original_data_path'] + '/' + cato
        ref_out = args['out_data_path'] + '/' + cato
        ref_split_out = args['out_data_path'] + '/' + cato + '/' + args['label_folder']

        if not os.path.isdir(ref_out):
            os.makedirs(ref_out)

        if not os.path.isdir(ref_split_out):
            os.makedirs(ref_split_out)

        for time in [i.split('.')[0] for i in sorted(os.listdir(data_ref))]:
            data = pd.read_csv(data_ref + '/' + time + '.csv', low_memory=False)
            ror_ebgm_data, theta_init = ror_ebgm(data, theta_init)
            ror_ebgm_data[(ror_ebgm_data['N'] >= 5) & ((ror_ebgm_data['N'] + ror_ebgm_data['c']) >= 300)].to_csv(ref_split_out + '/' + time + '.csv', index=False, header=True)

# Model: DySPred

args_model = {'base_path': './Data',
              'origin_path': 'all',
              'core_folder': 'Cores',
              'label_folder': 'split_year',
              'learn_type': 'regression',
              'model_type': ['DySPred'],
              'dic_level': 'PT',
              'duration': 5,
              'train': True,
              'seed': 20,
              'embed_dim': 128,
              'use_cuda': True,
              'min_epoch': 1000,
              'max_epoch': 10000,
              'patient': 10,
              'lr': 0.001,
              'export': True,
              'min_N': 10,
              'trans_layer_num': 1,
              'diffusion_layer_num': 1,
              'rnn_type': 'LSTM',
              'trans_activate_type': 'N',
              'bias': True,
              'weight_decay': 1e-3,
              'neg_num': 20,
              'cls_file': "cls_01",
              'cls_bias': True,
              'cls_layer_num': 2,
              'cls_output_dim': 5,
              'cls_activate_type': 'N',
              'train_ratio': 0.8,
              'val_ratio': 0.1,
              'test_ratio': 0.1,
              'test_stamp_idx': [0, -1],
              'has_cuda': True if torch.cuda.is_available() else False
              }
def model(args):
    f_list = [i.split('.')[0] for i in sorted(os.listdir(args['base_path'] + '/' + args['origin_path'] + '/' + args['label_folder']), reverse=True)]
    args['f_pre'] = f_list[0:(0+args['duration'])]
    Edge_Regression(args)

if __name__ == '__main__':
    # preprocessing(args_preprocessing)
    model(args_model)