from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pytorch_classification.utils.eval import AverageMeter, accuracy
from torchinfo import summary

# For tensorboard
from torch.utils.tensorboard import SummaryWriter
import torchvision


""" The training loop.
Input: 
    - trainset: train data with loader for the training, 
    - model: the network model, 
    - criterion: the loss criterion, 
    - optimizer: the optimizer used, 
    - epoch: the epoch used, 
    - use_cuda: use the GPU or not, 
    - writer: the tensorboard logger 
"""
def train(trainset, model, criterion, optimizer, epoch, use_cuda, writer, args):
    """ Use training mode"""
    model.train()

    """ Averaged training estimates"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()


    """ Loop over the training data """        
    for batch_idx, (inputs, targets) in enumerate(trainset.loader):
        """ Measure data loading time """
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        """ Get predictions and loss"""
        outputs = model(inputs)
        loss = criterion(outputs, targets) + args.weight_decay * model.module.extra_reg

        """ Measure accuracy and loss """
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5)) 
        losses.update(loss.data.cpu().detach().numpy(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        """ Backward pass"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """ Measure the time for 1 training step"""
        batch_time.update(time.time() - end)
        end = time.time()

        print("({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | "+\
            "Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}"\
            .format(
                batch=batch_idx + 1,
                size=len(trainset.loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg))
        torch.cuda.empty_cache() 
    return (losses.avg, top1.avg)


""" Running the test loop.
Input:     
    - testset: test set with data loader,
    - model: the networks 
    - criterion: the loss criterion, 
    - epoch: the current epoch number, 
    - use_cuda: on GPU or not
"""
def test(testset, model, criterion, epoch, use_cuda, args):
    """ Set the model to evaluate mode """
    model.eval()

    """ Averaged training estimates"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    """ Loop over the test data batches"""
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testset.loader):
        """ Measure the data loading time """
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        """ Get network predictions and loss"""
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        """ Estimate accuracy and loss """
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.cpu().detach().numpy(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        """ Measure time per batch at test time """
        batch_time.update(time.time() - end)
        end = time.time()

        print("({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | "+\
            "Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}"\
            .format(
                batch=batch_idx + 1,
                size=len(testset.loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg))
        torch.cuda.empty_cache() 
    return (losses.avg, top1.avg)

  
