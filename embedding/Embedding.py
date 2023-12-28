import gc
import os
import time
import copy
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from utils import check_and_make_path, sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

class SupervisedBase():
    def __init__(self, adj_list, idx_train, label_train, idx_val, label_val, idx_test, label_test, x_list, target_path,
                 embedding_folder, f_pre, node_list, model, loss, down_task, model_folder, model_file, model_type,
                 down_task_file, load_model, export, has_cuda, suffix):

        self.model_type = model_type
        self.target_path = target_path
        self.embedding_path = os.path.abspath(os.path.join(target_path, embedding_folder, model_file + model_type + suffix))
        self.model_path = os.path.abspath(os.path.join(target_path, model_folder, model_file + model_type + suffix))
        self.down_task_path = os.path.abspath(os.path.join(target_path, model_folder, down_task_file + model_type + suffix))
        self.predict_folder = os.path.abspath(os.path.join(target_path, 'Predict'))
        self.test_folder = os.path.abspath(os.path.join(target_path, 'Test'))

        self.timestamp_list = sorted(f_pre)
        self.has_cuda = has_cuda
        self.device = torch.device('cuda') if has_cuda else torch.device('cpu')

        self.model = model
        self.down_task = down_task
        self.loss = loss
        self.node_list = node_list
        self.node_num = len(node_list)
        self.load_model = load_model
        self.export = export

        check_and_make_path(self.embedding_path)
        check_and_make_path(self.model_path)
        check_and_make_path(self.predict_folder)
        check_and_make_path(self.test_folder)

        self.model_file = os.path.join(self.model_path, model_file)
        self.down_task_file = os.path.join(self.model_path, down_task_file)

        self.idx_train, self.label_train = idx_train, label_train
        self.idx_val, self.label_val = idx_val, label_val
        self.idx_test, self.label_test = idx_test, label_test
        self.x_list, self.adj_list = x_list, adj_list

    def clear_cache(self):
        if self.has_cuda:
            torch.cuda.empty_cache()
        else:
            gc.collect()

    def prepare(self, lr=1e-3, weight_decay=0.0):

        if self.load_model:
            if os.path.exists(self.model_file):
                self.model.load_state_dict(torch.load(self.model_file))
                self.model.eval()
                self.down_task.load_state_dict(torch.load(self.down_task_file))
                self.down_task.eval()

        self.model = self.model.to(self.device)
        self.down_task = self.down_task.to(self.device)
        self.loss = self.loss.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.8, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

        # scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
        # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

        optimizer.zero_grad()

        return self.model, self.down_task, self.loss, optimizer

    def get_test_info(self, edge_labels):
        idx_test, label_test = [], []
        test_items, test_labels = edge_labels[:, :2], edge_labels[:, 2]
        for i in range(len(self.timestamp_list)):
            idx_test.append(test_items)
            label_test.append(test_labels)
        return idx_test, label_test

    def get_predict_info(self, edge_labels):
        idx_predict = []
        predict_items = edge_labels
        for i in range(len(self.timestamp_list)):
            idx_predict.append(predict_items)
        return idx_predict

    def get_model_res(self, adj_list, x_list, batch_indices, model, down_task, loss_model=None, label=None):
        if self.model_type in ['DySPred_W', 'DySPred', 'DySGCN', 'DySGCN_W']:
            embedding_list = model(x_list, adj_list)
            cls_list = down_task(embedding_list, batch_indices)
            if loss_model is None:
                return embedding_list[-1], cls_list[-1]
            else:
                return loss_model(cls_list, label), embedding_list[-1]

        if self.model_type in ['MLP']:
            embedding_list = model(adj_list[-1][0])
            cls_list = down_task(embedding_list, batch_indices[-1])
            if loss_model is None:
                return embedding_list, cls_list
            else:
                return loss_model(cls_list, label[-1]), embedding_list

        if self.model_type in ['DySPred_Without_LSTM']:
            embedding_list = model(x_list[-1], adj_list[-1])
            cls_list = down_task(embedding_list, batch_indices[-1])
            if loss_model is None:
                return embedding_list, cls_list
            else:
                return loss_model(cls_list, label[-1]), embedding_list

    def save_embedding(self, output_list):
        if isinstance(output_list, torch.Tensor) and len(output_list.size()) == 2:
            embedding = output_list
            output_list = [embedding]

        embedding = output_list[-1]
        timestamp = self.timestamp_list[-1]
        df_export = pd.DataFrame(data=embedding.cpu().detach().numpy(), index=self.node_list)
        embedding_path = os.path.join(self.embedding_path, timestamp + '.csv')
        df_export.to_csv(embedding_path, header=True, index=True)

class SupervisedRegression(SupervisedBase):

    def __init__(self, adj_list, idx_train, label_train, idx_val, label_val, idx_test, label_test, x_list, target_path,
                 embedding_folder, f_pre, node_list, model, loss, down_task, model_folder='model', model_file='ctgcn',
                 model_type='', down_task_file='ctgcn_cls', load_model=False, export=True, has_cuda=False, suffix=''):
        super(SupervisedRegression, self).__init__(adj_list, idx_train, label_train, idx_val, label_val, idx_test,
                                                       label_test, x_list, target_path, embedding_folder, f_pre,
                                                       node_list, model, loss, down_task, model_folder, model_file,
                                                       model_type, down_task_file, load_model, export, has_cuda, suffix)


    def train(self, min_epoch=50, max_epoch=50, patient=200, lr=1e-3, weight_decay=0.):
        # prepare model, loss model, optimizer and classifier model
        model, down_task, loss_model, optimizer = self.prepare(lr, weight_decay)
        optimizer.add_param_group({'params': loss_model.noise_sigma, 'name': 'noise_sigma'})
        self.clear_cache()

        best_loss, best_noise_sigma, idx_patient = np.inf, np.inf, 0
        model.train()

        # scaler = torch.cuda.amp.GradScaler()

        for i in range(max_epoch):
            t1 = time.time()

            loss_train, _ = self.get_model_res(self.adj_list, self.x_list, self.idx_train, model, down_task, loss_model, self.label_train)

            # validation
            with torch.no_grad():
                loss_val, _ = self.get_model_res(self.adj_list, self.x_list, self.idx_val, model, down_task, loss_model, self.label_val)

                print('Epoch: ' + str(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()),
                      'noise_sigma: {:.4f}'.format(loss_model.noise_sigma.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'cost time: {:.4f}s'.format(time.time() - t1),
                      'lr: {:.5f}'.format(optimizer.param_groups[0]['lr']))

            if idx_patient >= patient and i >= min_epoch:
                break
            else:
                idx_patient += 1

            if i < min_epoch:
                idx_patient = 0

            if loss_val < best_loss and loss_val != -np.inf:
                best_loss = loss_val
                best_noise_sigma = loss_model.noise_sigma.item()
                idx_patient = 0
                if self.model_file:
                    torch.save(model.state_dict(), self.model_file)
                if self.down_task_file:
                    torch.save(down_task.state_dict(), self.down_task_file)
            self.clear_cache()

            # scaler.scale(loss_train).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss_train.backward()
            optimizer.step()

            model.zero_grad()

        print('finish supervised training!', 'best_loss: {:.4f}'.format(best_loss), 'best_noise_sigma: {:.4f}'.format(best_noise_sigma), 'loss_val: {:.4f}'.format(loss_val))

        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        model.eval()
        down_task = self.down_task.to(self.device)
        down_task.load_state_dict(torch.load(self.down_task_file, map_location=self.device))
        down_task.eval()

        with torch.no_grad():
            output_list, _ = self.get_model_res(self.adj_list, self.x_list, self.idx_val, model, down_task)

        if self.export:
            self.save_embedding(output_list)
        del output_list, model
        self.clear_cache()


    def test(self, edge_label):
        adj_list, x_list = self.adj_list, self.x_list

        test_model = self.model.to(self.device)
        test_model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        test_model.eval()
        test_down_task = self.down_task.to(self.device)
        test_down_task.load_state_dict(torch.load(self.down_task_file, map_location=self.device))
        test_down_task.eval()
        loss_model = self.loss.to(self.device)

        idx_test, label_test = self.get_test_info(edge_label)

        with torch.no_grad():
            loss, _ = self.get_model_res(adj_list, x_list, idx_test, test_model, test_down_task, loss_model, label_test)
        return loss

    def predict(self, edge_list):
        adj_list, x_list = self.adj_list, self.x_list

        predict_model = self.model.to(self.device)
        predict_model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        predict_model.eval()
        predict_down_task = self.down_task.to(self.device)
        predict_down_task.load_state_dict(torch.load(self.down_task_file, map_location=self.device))
        predict_down_task.eval()
        # loss_model = self.loss.to(self.device)

        idx_predict = self.get_predict_info(edge_list)

        with torch.no_grad():
            _, loss_input_list = self.get_model_res(adj_list, x_list, idx_predict, predict_model, predict_down_task)

        preds = loss_input_list.cpu().detach().numpy()
        results = pd.DataFrame(np.hstack((edge_list.cpu().numpy(), preds))).set_axis(['Drug', 'Reac', 'Score'], axis=1)

        idx2node_dict = dict(zip(np.arange(self.node_num), self.node_list))
        results[['Drug', 'Reac']] = results[['Drug', 'Reac']].applymap(lambda x: idx2node_dict[x])

        return results