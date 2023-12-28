import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma=2.):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def bmc_loss(self, pred, target, noise_var):

        logits = - (pred - target.unique()).pow(2) / (2 * noise_var)
        one_hot = (target.reshape(-1, 1) == target.unique()).sum(dim=0)
        log_soft_out = logits - torch.log((logits.exp() * one_hot).sum(1).reshape(-1, 1))
        loss = F.nll_loss(log_soft_out, target - 1)
        loss = loss * (2 * noise_var).detach()

        return loss

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2

        if type(pred) == list:
            loss_total = 0
            for i in range(len(pred)):
                loss = self.bmc_loss(pred[i], target[i], noise_var)
                loss_total += loss
            loss = loss_total / len(pred)
        else:
            loss = self.bmc_loss(pred, target, noise_var)
        # print('noise: {:.4f}'.format(noise_var))

        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def mse_loss(self, pred, target):

        logits = -(pred - target.unique()).pow(2)
        one_hot = (target.reshape(-1, 1) == target.unique()).sum(dim=0)

        loss = F.nll_loss(logits*one_hot.sum()/one_hot, target - 1)

        return loss

    def forward(self, pred, target):

        if type(pred) == list:
            loss_total = 0
            for i in range(len(pred)):
                loss = self.mse_loss(pred[i], target[i])
                loss_total += loss
            loss = loss_total / len(pred)
        else:
            loss = self.bmc_loss(pred, target)
        return loss



class MultiBMCLoss(nn.Module):
    def __init__(self):
        super(MultiBMCLoss, self).__init__()

    def Multi_bmc_loss(self, pred, target):

        pred = torch.softmax(pred, 1)
        one_hot = (target.reshape(-1, 1) == torch.tensor([i+1 for i in range(pred.shape[1])], device=target.device)).sum(dim=0)
        one_hot = one_hot / one_hot.sum()
        log_soft_out = torch.log(pred.exp() * one_hot / (pred.exp() * one_hot).sum(1).reshape(-1, 1))
        loss = F.nll_loss(log_soft_out, target - 1)

        return loss

    def forward(self, pred, target):
        # noise_var = self.noise_sigma ** 2

        if type(pred) == list:
            loss_total = 0
            for i in range(len(pred)):
                loss = self.Multi_bmc_loss(pred[i], target[i])
                loss_total += loss
            loss = loss_total / len(pred)
        else:
            loss = self.Multi_bmc_loss(pred, target)

        return loss


class ClassificationLoss(nn.Module):

    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def forward(self, cls_res, labels_list):
        cross_loss = nn.CrossEntropyLoss()

        if type(cls_res) == list:
            loss_total = 0
            for i in range(len(cls_res)):
                loss = cross_loss(cls_res[i], labels_list[i] - 1)
                loss_total += loss
            loss = loss_total / len(cls_res)
        else:
            loss = cross_loss(cls_res, labels_list - 1)
        return loss

class RegressionLoss(nn.Module):

    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, cls_res, labels_list):
        mse_loss = nn.MSELoss(reduction='sum')

        if type(cls_res) == list:
            loss_total = 0
            for i in range(len(cls_res)):
                preds = cls_res[i]
                labels = labels_list[i]
                loss = mse_loss(preds.squeeze(), labels.float()) / preds.shape[0]
                loss_total += loss
            loss = loss_total / len(cls_res)
        else:
            preds = cls_res
            loss = mse_loss(preds.squeeze(), labels_list.float()) / preds.shape[0]

        return loss

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)
        self.gamma = gamma

    def f_loss(self, preds, labels):
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())

        return loss.mean()

    def forward(self, pred, target):

        if type(pred) == list:
            loss_total = 0
            for i in range(len(pred)):
                loss = self.f_loss(pred[i], target[i]-1)
                loss_total += loss
            loss = loss_total / len(pred)
        else:
            loss = self.self.f_loss(pred, target-1)

        return loss