import os
import json
import torch
import sklearn
import scipy.sparse as sp
import numpy as np
import pandas as pd
import networkx as nx
from utils import sparse_mx_to_torch_sparse_tensor


class DataLoader:
    def __init__(self, file_path, node_list, reac_node_list, f_pre, known_edges, has_cuda=False, RxNorm=None, MedDRA=None):
        self.file_path = file_path
        self.f_pre = f_pre
        self.node_list = node_list
        self.reac_node_list = reac_node_list
        self.node_num = len(self.node_list)
        self.node2idx_dict = dict(zip(self.node_list, np.arange(self.node_num)))
        self.has_cuda = has_cuda
        known_edges[['Reac', 'Drug']] = known_edges[['Reac', 'Drug']].applymap(lambda x: self.node2idx_dict[x])
        known_edges = known_edges.set_axis(['from_id', 'to_id', 'signal'], axis=1)
        self.known_edges = known_edges
        self.RxNorm = RxNorm
        self.MedDRA = MedDRA

    def get_core_adj_list(self, core):
        date_dir_list = sorted(self.f_pre)
        core_adj_list = []

        for i in date_dir_list:
            data = pd.read_csv(self.file_path + '/' + i + '.csv')
            data = data[data['Reac'].isin(self.node_list) & data['Drug'].isin(self.node_list)]

            graph = self.get_nx_graph(data, self.node_list)
            core_num_dict = nx.core_number(graph)
            step = int((np.median([i for i in core_num_dict.values() if i != 0]) - 20) / 2)
            core_list = [20 + i*step for i in range(5)]
            core_list = core_list[:core]

            tmp_adj_list = []
            for c in core_list[::-1]:
                k_core_graph = nx.k_core(graph, k=c, core_number=core_num_dict)
                k_core_graph.add_nodes_from(self.node_list)
                spmat = nx.to_scipy_sparse_matrix(k_core_graph, nodelist=self.node_list)
                spmat = sklearn.preprocessing.normalize(spmat, norm='l2')
                spmat = spmat + sp.eye(spmat.shape[0])
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                tmp_adj_list.append(sptensor.cuda() if self.has_cuda else sptensor)
            core_adj_list.append(tmp_adj_list)
        return core_adj_list

    @staticmethod
    def get_nx_graph(data, node_list, graph_type='Directed'):

        df = data.iloc[:, :5].set_axis(['from', 'to', 'label', 'weight1', 'weight2'], axis=1)
        if graph_type == 'Unirected':
            df['weight'] = np.log(np.sqrt(df['weight1'] * df['weight2']) + 1)
            graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='weight', create_using=nx.Graph)
        else:
            data = pd.concat([df[['from', 'to', 'weight1']].set_axis(['from', 'to', 'weight'], axis=1),
                              df[['from', 'to', 'weight2']].set_axis(['to', 'from', 'weight'], axis=1)])
            graph = nx.from_pandas_edgelist(data, 'from', 'to', edge_attr='weight', create_using=nx.DiGraph)
        graph.add_nodes_from(np.arange(len(node_list)))
        return graph

    def get_feature_list(self, pre=False, dic_level='PT'):
        if pre == True:
            # Drug
            temp = self.RxNorm[['concept_name_1', 'concept_code_2']].set_axis(['Drug', 'ATC'], axis=1)
            temp = pd.concat(
                [pd.concat([temp['Drug'], temp['ATC'] + '/' + temp['Drug']], axis=1).set_axis(['Drug', 'ATC'], axis=1),
                 pd.concat([temp['Drug'], temp['ATC'].str[:1]], axis=1),
                 pd.concat([temp['Drug'], temp['ATC'].str[:3]], axis=1),
                 pd.concat([temp['Drug'], temp['ATC'].str[:4]], axis=1),
                 pd.concat([temp['Drug'], temp['ATC'].str[:5]], axis=1),
                 temp], axis=0)

            def weight_set(x, delta=0.5):
                if len(x) > 7:
                    return 1
                elif len(x) == 7:
                    return delta
                elif len(x) == 5:
                    return delta ** 2
                elif len(x) == 4:
                    return delta ** 3
                elif len(x) == 3:
                    return delta ** 4
                else:
                    return delta ** 5

            temp['weight'] = temp.apply(lambda x: weight_set(x['ATC']), axis=1)
            temp = temp.drop_duplicates()
            CT = temp[['Drug', 'weight']].groupby('Drug').sum('weight').reset_index().rename(columns={'Drug': 'Drug_y'})
            temp = temp[temp['Drug'].isin(self.node_list)]

            node_item = sorted(temp['Drug'].unique())
            DAG_Drug = pd.DataFrame()
            for i in range(int(len(node_item) / 2 + 1)):
                data1 = temp[temp['Drug'] == node_item[i]]
                weight_i = CT.loc[CT['Drug_y'] == node_item[i], 'weight'].values
                w = data1.merge(temp, on='ATC', how='inner').merge(CT, on='Drug_y', how='inner')
                w = w[['Drug_y', 'Drug_x', 'weight_y', 'weight']].groupby(['Drug_y', 'Drug_x', 'weight']).sum(
                    'weight_y').reset_index()
                w['SS'] = w['weight_y'] * 2 / (weight_i + w['weight'])
                DAG_Drug = pd.concat([DAG_Drug, w[['Drug_x', 'Drug_y', 'SS']].drop_duplicates().set_axis(['Node1', 'Node2', 'SS'], axis=1)])

            # DAG_Drug = DAG_Drug[DAG_Drug['Node1'] == DAG_Drug['Node2']]

            # Reac
            if dic_level == 'PT':
                temp = self.MedDRA.loc[self.MedDRA['descendant_class_id'] == 'PT',
                                       ['ancestor_name', 'ancestor_class_id', 'descendant_name']].set_axis(['Class', 'Class_id', 'PT'], axis=1)
                temp['PT'] = temp['PT'].str.upper()

                def weight_set(x, delta=0.6):
                    if x == 'PT':
                        return 1
                    elif x == 'HLT':
                        return delta
                    elif x == 'HLGT':
                        return delta ** 2
                    else:
                        return delta ** 3

                temp['weight'] = temp.apply(lambda x: weight_set(x['Class_id']), axis=1)
                temp = temp.drop_duplicates()
                CT = temp[['PT', 'weight']].groupby('PT').sum('weight').reset_index().rename(columns={'PT': 'PT_y'})
                temp = temp[temp['PT'].isin(self.node_list)]

                node_item = sorted(temp['PT'].unique())
                DAG_Reac = pd.DataFrame()
                for i in range(int(len(node_item) / 2 + 1)):
                    data1 = temp[temp['PT'] == node_item[i]]
                    weight_i = CT.loc[CT['PT_y'] == node_item[i], 'weight'].values
                    w = data1.merge(temp, on='Class', how='inner').merge(CT, on='PT_y', how='inner')
                    w = w[['PT_y', 'PT_x', 'weight_y', 'weight']].groupby(['PT_y', 'PT_x', 'weight']).sum('weight_y').reset_index()
                    w['SS'] = w['weight_y'] * 2 / (weight_i + w['weight'])
                    DAG_Reac = pd.concat([DAG_Reac, w[['PT_x', 'PT_y', 'SS']].drop_duplicates().set_axis(['Node1', 'Node2', 'SS'], axis=1)])

            if dic_level == 'HLT':
                temp = self.MedDRA.loc[self.MedDRA['descendant_class_id'] == 'HLT',
                                       ['ancestor_name', 'ancestor_class_id', 'descendant_name']].set_axis(
                    ['Class', 'Class_id', 'HLT'], axis=1)
                temp['HLT'] = temp['HLT'].str.upper()

                def weight_set(x, delta=0.6):
                    if x == 'HLT':
                        return 1
                    elif x == 'HLGT':
                        return delta
                    else:
                        return delta ** 2

                temp['weight'] = temp.apply(lambda x: weight_set(x['Class_id']), axis=1)
                temp = temp.drop_duplicates()
                CT = temp[['HLT', 'weight']].groupby('HLT').sum('weight').reset_index().rename(columns={'HLT': 'HLT_y'})
                temp = temp[temp['HLT'].isin(self.node_list)]

                node_item = sorted(temp['HLT'].unique())
                DAG_Reac = pd.DataFrame()
                for i in range(int(len(node_item) / 2 + 1)):
                    data1 = temp[temp['HLT'] == node_item[i]]
                    weight_i = CT.loc[CT['HLT_y'] == node_item[i], 'weight'].values
                    w = data1.merge(temp, on='Class', how='inner').merge(CT, on='HLT_y', how='inner')
                    w = w[['HLT_y', 'HLT_x', 'weight_y', 'weight']].groupby(['HLT_y', 'HLT_x', 'weight']).sum(
                        'weight_y').reset_index()
                    w['SS'] = w['weight_y'] * 2 / (weight_i + w['weight'])
                    DAG_Reac = pd.concat([DAG_Reac,
                                          w[['HLT_x', 'HLT_y', 'SS']].drop_duplicates().set_axis(['Node1', 'Node2', 'SS'],
                                                                                               axis=1)])

            # tocoo
            spmat = sp.lil_matrix((len(self.node_list), len(self.node_list)))
            DAG = pd.concat([DAG_Drug, DAG_Reac])
            DAG[['Node1', 'Node2']] = DAG[['Node1', 'Node2']].applymap(lambda x: self.node2idx_dict[x])

            for line in DAG.values:
                spmat[int(line[0]), int(line[1])] = line[2]
                spmat[int(line[1]), int(line[0])] = line[2]
                spmat[int(line[0]), int(line[0])] = 1
                spmat[int(line[1]), int(line[1])] = 1

            spmat = spmat.tocoo()
        else:
            spmat = sp.eye(self.node_num)
        sptensor = sparse_mx_to_torch_sparse_tensor(spmat)

        x_list = []
        for i in range(len(self.f_pre)):
            x_list.append(sptensor.cuda() if self.has_cuda else sptensor)

        return x_list

    def get_edge_label_list_train(self, label_path, label_file_list, min_N=None):
        label_file_list = sorted(label_file_list)
        edge_label_list = []
        for i in range(len(label_file_list)):
            label_file_path = os.path.join(label_path, label_file_list[i] + '.csv')
            df_edges = pd.read_csv(label_file_path)
            df_edges = df_edges.loc[:, df_edges.columns[:6]]
            df_edges.columns = ['from_id', 'to_id', 'label', 'weight_1', 'weight_2', 'N']
            df_edges = df_edges[df_edges['from_id'].isin(self.node_list) & df_edges['to_id'].isin(self.node_list)]
            if min_N is not None:
                df_edges = df_edges[df_edges['N'] >= min_N]
            df_edges[['from_id', 'to_id']] = df_edges[['from_id', 'to_id']].applymap(lambda x: self.node2idx_dict[x])
            edge_label_list.append(df_edges)
        return edge_label_list

    def get_edge_label_list_test(self, label_path, label_file):

        label_file_path = os.path.join(label_path, label_file)
        df_edges = pd.read_csv(label_file_path)
        df_edges = df_edges.loc[(df_edges['Drug'].isin(self.node2idx_dict.keys())) &
                                (df_edges['Reac'].isin(self.node2idx_dict.keys())), df_edges.columns[:3]]
        df_edges.columns = ['from_id', 'to_id', 'label']
        df_edges[['from_id', 'to_id']] = df_edges[['from_id', 'to_id']].applymap(lambda x: self.node2idx_dict[x])

        return torch.from_numpy(df_edges.values).long().cuda() if self.has_cuda else torch.from_numpy(df_edges.values).long()

    def get_predict_list(self, drug_list):
        predict_list = [[i, j]for i in drug_list for j in self.reac_node_list]
        predict_list = pd.DataFrame(predict_list).set_axis(['from_id', 'to_id'], axis=1).applymap(lambda x: self.node2idx_dict[x])
        return torch.from_numpy(predict_list.values).long().cuda() if self.has_cuda else torch.from_numpy(predict_list.values).long()

    def tensor_transfer(self, data):
        data = [torch.from_numpy(i.values).cuda() if self.has_cuda else torch.from_numpy(i.values) for i in data]
        return data