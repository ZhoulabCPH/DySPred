import os
import pandas as pd
import numpy as np
import random
import torch
import copy
import networkx as nx
import sklearn
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from embedding.DataLoader import DataLoader
from embedding.DySPred import DySGCN, DySPred, MLP, DySPred_Without_LSTM
from embedding.baseline.gcrn import GCRN
from embedding.Loss import BMCLoss, MultiBMCLoss, ClassificationLoss, RegressionLoss, Focal_loss
from embedding.Embedding import SupervisedRegression
from embedding.Classifier import EdgeClassifier
from utils import sparse_mx_to_torch_sparse_tensor

class Base_task():
    def __init__(self, args):
        self.args = args
        self.setup_seed(args['seed'])
        self.base_path = args['base_path']
        self.origin_path = args['origin_path']
        self.core_folder = args['core_folder']
        self.label_folder = args['label_folder']
        self.learn_type = args['learn_type']

        self.train = args['train']
        self.embed_dim = args['embed_dim']
        self.model_file = self.learn_type
        self.f_pre = sorted(args['f_pre'])
        self.duration = len(self.f_pre)
        self.has_cuda = args['has_cuda'] & args['use_cuda']
        self.min_epoch = args['min_epoch']
        self.max_epoch = args['max_epoch']
        self.patient = args['patient']
        self.lr = args['lr']
        self.load_model = False if args['train'] == True else True
        self.export = args['export']
        self.min_N = args['min_N']
        self.weight_decay = args['weight_decay']
        self.train_ratio = args['train_ratio']
        self.val_ratio = args['val_ratio']
        self.test_ratio = args['test_ratio']
        self.cls_file = args['cls_file']
        self.test_stamp_idx = args['test_stamp_idx']
        self.label_path = os.path.abspath(os.path.join(self.base_path, self.origin_path, self.label_folder))
        self.target_path = os.path.abspath(os.path.join(self.base_path, self.origin_path, self.f_pre[-1]))

        self.data_loader = self.get_data_loader()
        self.node_list = self.data_loader.node_list
        self.node2idx = self.data_loader.node2idx_dict
        self.reac_node_list = self.data_loader.reac_node_list

        self.adj_list, self.idx_train, self.label_train, self.idx_val, self.label_val, self.idx_test, self.label_test, self.train_all_items = self.get_train_val_test_info()

        init = pd.Series(self.edge_label_list[-1][:, 2].unique().cpu())
        for i in range(self.duration):
            init = init + pd.Series(self.edge_label_list[i][:, 2].cpu() - 1).value_counts()
        self.alpha_init = (init.sum() / init).to_list()

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    def get_data_loader(self):
        data = pd.read_csv(self.label_path + '/' + self.f_pre[-1] + '.csv')
        data = data[data['N'] >= self.min_N]
        full_node_list = sorted(list(set(data.iloc[:, 0].to_list() + data.iloc[:, 1].to_list())))
        reac_node_list = sorted(list(set(data.iloc[:, 1].to_list())))
        node_num = len(full_node_list)

        RxNorm = pd.read_csv(self.base_path + '/' + 'ATC_RxNorm.csv')
        MedDRA = pd.read_csv(self.base_path + '/' + 'ATC_MedDRA.csv')

        data_loader = DataLoader(self.label_path, full_node_list, reac_node_list, self.f_pre, data.loc[:, ['Drug', 'Reac', 'signal']], self.has_cuda, RxNorm, MedDRA)
        core_base_path = os.path.abspath(os.path.join(self.target_path, self.core_folder)) if self.core_folder else None

        self.core_base_path = core_base_path
        self.node_num = node_num
        self.input_dim = node_num

        return data_loader

    def get_train_val_test_info(self):
        edge_label_list = self.data_loader.get_edge_label_list_train(self.label_path, self.f_pre, self.min_N)
        self.edge_label_list = self.data_loader.tensor_transfer(edge_label_list)

        timestamp_num = len(self.edge_label_list)

        X_train, X_test, y_train, y_test = train_test_split(edge_label_list[-1],
                                                            np.arange(edge_label_list[-1].shape[0]),
                                                            test_size=self.test_ratio + self.val_ratio,
                                                            stratify=edge_label_list[-1]['label'],
                                                            random_state=self.args['seed'])
        X_val, X_test, y_val, y_test = train_test_split(X_test,
                                                        y_test,
                                                        test_size=self.test_ratio / (self.test_ratio + self.val_ratio),
                                                        stratify=X_test['label'],
                                                        random_state=self.args['seed'])
        val_pair = self.edge_label_list[-1][y_val, :2]
        test_pair = self.edge_label_list[-1][y_test, :2]

        adj_list, idx_train, label_train, idx_val, label_val, idx_test, label_test, train_all_items = [], [], [], [], [], [], [], []

        for i in range(timestamp_num):

            val_idx = [True in ((item == val_pair).sum(axis=1) >= 2) for item in self.edge_label_list[i][:, :2]]
            test_idx = [True in ((item == test_pair).sum(axis=1) >= 2) for item in self.edge_label_list[i][:, :2]]
            train_idx = [(val_idx[k] + test_idx[k]) == 0 for k in range(len(val_idx))]

            train = self.edge_label_list[i][train_idx]
            val = self.edge_label_list[i][val_idx]
            test = self.edge_label_list[i][test_idx]

            graph = self.data_loader.get_nx_graph(pd.DataFrame(train.detach().cpu().numpy()), self.node_list)

            core_num_dict = nx.core_number(graph)
            core_list = list(np.percentile([i for i in core_num_dict.values() if i != 0], [0, 25, 50, 75, 100]))

            tmp_adj_list = []
            for c in core_list[::-1]:
                k_core_graph = nx.k_core(graph, k=c, core_number=core_num_dict)
                k_core_graph.add_nodes_from(np.arange(len(self.node_list)))
                spmat = nx.to_scipy_sparse_matrix(k_core_graph)
                spmat = sklearn.preprocessing.normalize(spmat, norm='l2')
                spmat = spmat + sp.eye(spmat.shape[0])
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                tmp_adj_list.append(sptensor.cuda() if self.has_cuda else sptensor)
            adj_list.append(tmp_adj_list)

            if i == (timestamp_num - 1):
                train_all_items = train[:, :2]

            train_items, train_labels = train[train[:, -1] >= self.min_N, :2], train[train[:, -1] >= self.min_N, 2].long()
            val_items, val_labels = val[val[:, -1] >= self.min_N, :2], val[val[:, -1] >= self.min_N, 2].long()
            test_items, test_labels = test[test[:, -1] >= self.min_N, :2], test[test[:, -1] >= self.min_N, 2].long()

            idx_train.append(train_items)
            label_train.append(train_labels)
            idx_val.append(val_items)
            label_val.append(val_labels)
            idx_test.append(test_items)
            label_test.append(test_labels)

        return adj_list, idx_train, label_train, idx_val, label_val, idx_test, label_test, train_all_items

    def get_dynamic_model(self, model_type):
        bias = self.args.get('bias', None)
        trans_num = self.args['trans_layer_num']
        diffusion_num = self.args['diffusion_layer_num']
        hidden_dim = self.embed_dim * 4
        rnn_type = self.args['rnn_type']
        trans_activate_type = self.args['trans_activate_type']

        if model_type == 'MLP':
            return MLP(self.input_dim, hidden_dim, self.embed_dim, 2)
        if model_type in ['DySPred_Without_LSTM', 'DySPred_Without_LSTM_W']:
            return DySPred_Without_LSTM(self.input_dim, hidden_dim, self.embed_dim, trans_num, diffusion_num, bias,
                         rnn_type, trans_activate_type)

        if model_type in ['DySPred', 'DySPred_W']:
            return DySPred(self.input_dim, hidden_dim, self.embed_dim, trans_num, diffusion_num, self.duration, bias,
                         rnn_type, trans_activate_type)
        if model_type in ['DySGCN', 'DySGCN_W']:
            return DySGCN(self.input_dim, hidden_dim, self.embed_dim, trans_num, diffusion_num, self.duration, bias,
                          rnn_type, trans_activate_type)

    def get_dynamic_loss(self, model_type, init_noise_sigma=2.0):
        cls_hidden_dim = int(self.embed_dim / 2)
        cls_layer_num = self.args.get('cls_layer_num', None)
        cls_output_dim = self.args.get('cls_output_dim', None)
        cls_bias = self.args.get('cls_bias', None)
        cls_activate_type = self.args.get('cls_activate_type', None)

        if self.learn_type == 'classification':
            classifier = EdgeClassifier(self.embed_dim, cls_hidden_dim, output_dim=cls_output_dim,
                                        layer_num=cls_layer_num,
                                        duration=self.duration, bias=cls_bias, activate_type=cls_activate_type)
            loss = MultiBMCLoss()
            # loss = Focal_loss(alpha=self.alpha_init)
        else:
            if model_type in ['MLP', 'DySPred_Without_LSTM']:
                classifier = EdgeClassifier(self.embed_dim, cls_hidden_dim, output_dim=1, layer_num=cls_layer_num,
                                            duration=1, bias=cls_bias, activate_type=cls_activate_type)
            if model_type in ['DySPred', 'DySPred_W', 'DySGCN', 'DySGCN_W']:
                classifier = EdgeClassifier(self.embed_dim, cls_hidden_dim, output_dim=1, layer_num=cls_layer_num,
                                            duration=self.duration, bias=cls_bias, activate_type=cls_activate_type)
            loss = BMCLoss(init_noise_sigma=init_noise_sigma)

        return loss, classifier

class Edge_Regression(Base_task):
    def __init__(self, args):
        super(Edge_Regression, self).__init__(args)
        self.args = args

        for model_t in args['model_type']:
            self.embedding_folder = model_t + '/embedding'
            self.model_folder = model_t + '/model'
            self.model_type = model_t

            # x_list model loss
            if model_t in ['DySPred', 'DySGCN_W', 'DySPred_Without_LSTM']:
                self.x_list = self.data_loader.get_feature_list(pre=True, dic_level=args['dic_level'])
            if model_t in ['DySPred_W', 'DySGCN_W', 'MLP']:
                self.x_list = self.data_loader.get_feature_list(pre=False, dic_level=args['dic_level'])

            model = self.get_dynamic_model(model_type=model_t)
            loss, classifier = self.get_dynamic_loss(model_type=model_t)

            # SupervisedRegression

            for core in range(5, 6, 1):

                self.model = copy.deepcopy(model)
                self.loss, self.classifier = copy.deepcopy(loss), copy.deepcopy(classifier)

                if model_t == 'MLP':
                    core = 1

                if model_t in ['DySPred', 'DySPred_Without_LSTM']:
                    suffix = '_True_' + str(args['seed']) + '_' + str(core) + \
                             '_' + str(args['trans_layer_num']) + '_' + str(args['diffusion_layer_num'])
                if model_t in ['DySPred_W', 'DySGCN_W']:
                    suffix = '_False_' + str(args['seed']) + '_' + str(core) + \
                             '_' + str(args['trans_layer_num']) + '_' + str(args['diffusion_layer_num'])
                if model_t == 'MLP':
                    suffix = '_' + str(args['seed'])

                adj_list = [adj[(-core):] for adj in self.adj_list]

                print('regression ' + args['origin_path'] + suffix + '_' + model_t)

                self.downstream_task = SupervisedRegression(adj_list, self.idx_train, self.label_train, self.idx_val,
                                                            self.label_val, self.idx_test, self.label_test, self.x_list,
                                                            self.target_path, self.embedding_folder, self.f_pre,
                                                            self.node_list, self.model, self.loss, self.classifier,
                                                            self.model_folder, self.model_file, self.model_type,
                                                            self.cls_file, self.load_model, self.export, self.has_cuda, suffix)



                if self.train:
                    self.downstream_task.train(self.min_epoch, self.max_epoch, self.patient, self.lr, self.weight_decay)

                ICIs_edge_predict = self.model_predict()
                ICIs_edge_test = self.model_test()

                ICIs_edge_predict.to_csv(os.path.join(self.downstream_task.predict_folder, args['origin_path'] + '_' + model_t + suffix + '.csv'), index=False)
                ICIs_edge_test.to_csv(os.path.join(self.downstream_task.test_folder, args['origin_path'] + '_' + model_t + suffix + '.csv'), index=False)

                if model_t == 'MLP':
                    break

    def model_test(self):

        list_ready = [i for i in sorted(os.listdir(self.label_path)) if i >= self.f_pre[-1]]
        test_label_file_list = list_ready
        results_out = pd.DataFrame()
        idx2node_dict = dict(zip(np.arange(self.data_loader.node_num), self.data_loader.node_list))

        train_data = pd.DataFrame(self.downstream_task.idx_train[-1].cpu().numpy()).set_axis(['Drug', 'Reac'], axis=1)
        train_data[['Drug', 'Reac']] = train_data[['Drug', 'Reac']].applymap(lambda x: idx2node_dict[x])
        train_data['data_type'] = 'train'

        val_data = pd.DataFrame(self.downstream_task.idx_val[-1].cpu().numpy()).set_axis(['Drug', 'Reac'], axis=1)
        val_data[['Drug', 'Reac']] = val_data[['Drug', 'Reac']].applymap(lambda x: idx2node_dict[x])
        val_data['data_type'] = 'val'

        test_data = pd.DataFrame(self.downstream_task.idx_test[-1].cpu().numpy()).set_axis(['Drug', 'Reac'], axis=1)
        test_data[['Drug', 'Reac']] = test_data[['Drug', 'Reac']].applymap(lambda x: idx2node_dict[x])
        test_data['data_type'] = 'test'

        data_split = pd.concat([train_data, val_data, test_data])

        if len(self.test_stamp_idx) > 0:
            for i in self.test_stamp_idx:
                test_edge_label = self.data_loader.get_edge_label_list_test(self.label_path, test_label_file_list[i])
                results_temp = self.downstream_task.predict(test_edge_label[:, :2])
                label_file_path = os.path.join(self.label_path, test_label_file_list[i])
                df_edges = pd.read_csv(label_file_path)
                results_temp = results_temp.merge(df_edges, on=['Drug', 'Reac'], how='left').merge(data_split, on=['Drug', 'Reac'], how='left')
                results_temp['Test_xM'] = str(i) + 'y'

                results_out = pd.concat([results_out, results_temp])

        return results_out

    def model_predict(self):
        results_out = pd.DataFrame()
        drug_list = [i for i in self.node_list if i not in self.reac_node_list]

        for i in range((len(drug_list) // 100)+1):
            # print(min(((i+1)*100), len(drug_list)))
            if (i * 100) != len(drug_list):
                predict_list = self.data_loader.get_predict_list(drug_list[(i*100):min(((i+1)*100), len(drug_list))])
                results = self.downstream_task.predict(predict_list)
                results_out = pd.concat([results_out, results])

        return results_out

    def model_predict_network(self):
        d_list = list(set(self.node_list) - set(self.data_loader.reac_node_list))

        results_out = pd.DataFrame()
        for i in range(len(d_list)//100 + 1):
            predict_list = self.data_loader.get_predict_list(d_list[i*100:min((i+1)*100, len(d_list))])
            results = self.downstream_task.predict(predict_list)
            results = results[results['Score'] == 3]
            results_out = pd.concat([results_out, results])
        return results_out
