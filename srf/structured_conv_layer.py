# Import general dependencies
import numpy as np
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Function
from torch.distributions import normal
from srf.gaussian_basis_filters import *

import torch.nn.functional as F
import time

""" The N-Jet convolutional-layer using a linear combination of 
Gaussian derivative filters.
Inputs:
    - inC: input channels
    - outC: output channels
    - init_k: the spatial extent of the kernels (default: 2)
    - init_order: the order of the approximation (default: 3)
    - init_scale: the initial starting scale, where: sigma=2^scale (default: 0)
    - learn_sigma: whether sigma is learnable
    - use_cuda: running on GPU or not
    - groups: groups for the convolution (default: 1)
    - ssample: if we subsample the featuremaps based on sigma (default: False)
"""
class Srf_layer_shared(nn.Module):
    def __init__(self,
                inC,
                outC, 
                init_k,
                init_order,
                init_scale,
                learn_sigma, 
                use_cuda,
                groups=1,
                ssample=False):
        super(Srf_layer_shared, self).__init__()

        self.init_k = init_k
        self.init_order = init_order
        self.init_scale = init_scale
        self.inC = inC
        self.ssample = ssample

        assert(outC % groups == 0)
        self.outC = outC
        self.groups = groups
 
        """ Define the number of basis based on order. """
        F = int((self.init_order + 1) * (self.init_order + 2) / 2)                        

        """ Create weight variables. """
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.alphas = torch.nn.Parameter(torch.zeros([F, int(inC/groups), outC], \
                            device=self.device), requires_grad=True) 


        """ Define the scale parameter. """
        torch.nn.init.normal_(self.alphas, mean=0.0, std=1)
        if learn_sigma:    
            self.scales = torch.nn.Parameter(torch.tensor(np.full((1), \
                            self.init_scale), device=self.device,\
                            dtype=torch.float32), requires_grad=True)      
        else:
            self.scales = torch.nn.Parameter(torch.tensor(np.full((1), \
                            self.init_scale), device=self.device,\
                            dtype=torch.float32), requires_grad=False)      
        self.extra_reg = 0

    """ Forward pass without inputs to return the filters only. """
    def forward_no_input(self):
        """ Define sigma from the scale: sigma = 2^scale """
        self.sigma = 2.0**self.scales
        self.filtersize = torch.ceil(self.init_k*self.sigma[0]+0.5)
        
        """ Define the grid on which the filter is created. """
        try:
            self.x = torch.arange(start=-self.filtersize.detach().cpu().float(), \
                    end=self.filtersize.detach().cpu().float()+1, step=1)
        except: 
            print("Sigma value is off:", self.sigma)


        """ Create the Gaussian derivative filters. """
        self.hermite = self.x
        self.filters, self.basis, self.gauss, self.hermite = gaussian_basis_filters_shared(
                                            x=self.x,\
                                            hermite=self.hermite,\
                                            order=self.init_order, \
                                            sigma=self.sigma, \
                                            alphas=self.alphas,\
                                            use_cuda=self.use_cuda)
        return self.filters


    """ Forward pass with inputs: creates the filters and performs the convolution. """
    def forward(self, data): 
        """ Define sigma from the scale: sigma = 2^scale """
        self.sigma = torch.pow(torch.tensor([2.0]).cuda(), self.scales)
        self.filtersize = torch.ceil(self.init_k*self.sigma[0]+0.5)
        
        """ Define the grid on which the filter is created. """
        try:
            
            self.x = torch.arange(start=-self.filtersize.detach().cpu().float(), \
                    end=self.filtersize.detach().cpu().float()+1, step=1)
        except: 
            print("Sigma value is off:", self.sigma, "filter size:", self.filtersize)


        """ Create the Gaussian derivative filters. """
        self.hermite = self.x
        self.filters, self.basis, self.gauss, self.hermite = gaussian_basis_filters_shared(
                                            x=self.x,\
                                            hermite=self.hermite,\
                                            order=self.init_order, \
                                            sigma=self.sigma, \
                                            alphas=self.alphas,\
                                            use_cuda=self.use_cuda)

        """ Subsample based on sigma if wanted. """
        if self.ssample:
            data = safe_sample(data, self.sigma)   

        """ Perform the convolution. """
        self.final_conv = F.conv2d(
                    input=data, # NCHW
                    weight=self.filters, # KCHW
                    bias=None,
                    stride=1,
                    padding=int(self.filters.shape[2]/2),
                    groups=self.groups)

        return self.final_conv

    """ List the parameters. """
    def num_params(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))


""" Subsampling of the featuremaps based on the learned sigma. 
Input:
    - current: input featuremap
    - sigma: the learned sigma values
    - r: the hyperparameter controlling how fast the subsampling goes as a function of sigma.
"""
def safe_sample(current, sigma, r=4.0):        
    update_val = max(1.0, torch.div(2**sigma, r))
    shape = current.shape
    shape_out = max([1,1], [int(float(shape[2])/update_val), \
                            int(float(shape[3])/update_val)])
    current_out = F.interpolate(current, shape_out)
    return current_out
