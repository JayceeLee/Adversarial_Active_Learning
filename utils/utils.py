import os
import sys

import numpy as np
import logging
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

def make_logger(filename, args):
    logger = logging.getLogger()
    file_log_handler = logging.FileHandler(os.path.join(args.exp_dir, filename))
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stderr_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter()
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    logger.info(args)
    return logger

def make_D_label(input, value, device, random=False):
    if random:
        if value == 0:
            lower, upper = 0, 0.305
        elif value == 1:
            lower, upper = 0.7, 1.05
        D_label = torch.FloatTensor(input.data.size()).uniform_(lower, upper).to(device)
    else:
        D_label = torch.FloatTensor(input.data.size()).fill_(value).to(device)
    return D_label

def make_shape_label(input, npts):
    device = input.device
    cls = torch.argmax(input, dim=1, keepdim=True)
    cls = cls.repeat(1,npts)
    cls = cls.long().to(device)
    return cls


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, learning_rate, i_iter, max_steps, power):
    lr = lr_poly(learning_rate, i_iter, max_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def get_semi_gt(D_outs, pred, pred_softmax, threshold, device):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > threshold:
            count += 1

    if count > 0:
        # print('> threshold: ', count, '/', D_outs.shape[0])
        pred_sel = torch.Tensor(count, pred.size(1), pred.size(2), pred.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0
        for j in range(D_outs.size(0)):
            if D_outs[j] > threshold:
                pred_sel[num_sel] = pred[j]
                semi_gt = torch.argmax(pred_softmax[j].data.cpu(), dim=0, keepdim=False)
                label_sel[num_sel] = semi_gt
                num_sel += 1
        return pred_sel.to(device), label_sel.long().to(device), count
    else:
        return 0, 0, count

def one_hot(label, device, num_classes=4):
    label = label.data.cpu().numpy()
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,i,...] = (label == i)
    return torch.FloatTensor(one_hot).to(device)

def convt_array_to_PIL(img):
    return Image.fromarray(img)

def rescale_image(img):
    min_v = img.min()
    max_v = img.max()
    img_scale = img[:,:]-min_v
    img_scale = img_scale / float(max_v-min_v)
    img_scale = 255 * img_scale
    img_scale = img_scale.astype(np.uint8)
    return img_scale