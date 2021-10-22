import torch
from scipy import ndimage
import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
    



""" Create the Gaussian derivative basis.
Input:
    - x: the input grid
    - hermite: a temporary variable (initialized as the grid x)
    - order: the order of the basis
    - sigma: the sigma of the Gaussian
    - alphas: the coefficients for combining the basis
    - use_cuda: on GPU or not
"""
def gaussian_basis_filters_shared(
        x,
        hermite,
        order,
        sigma,
        alphas, 
        use_cuda):
    basis_filters = []
    basis_tensors = []
            
    """ Define the 0th order Gaussian vector for the current scale. """
    if use_cuda: x = x.cuda()   
    gauss = torch.div(1.0, (math.sqrt(2.0 * math.pi) * sigma[0])) \
        * torch.exp( torch.div( x*x, (-2.0*sigma[0]*sigma[0])) )
    gauss = gauss / torch.sum(gauss)

    """ Define the rest of the Gaussian derivatives. """
    basis = []
    for i in range(0, int(order)+1):
        basis_x, hermite = get_basis(x, i, gauss, sigma[0], hermite)
        basis_x = torch.pow(sigma[0], i) * basis_x

        for j in range(int(order)-i, -1, -1):
            basis_y, hermite = get_basis(x, j, gauss, sigma[0], hermite)
            basis_y = torch.pow(sigma[0], j) * basis_y 

            """ Create square filters from the 1D vectors. """
            basis.append(torch.einsum("i,j->ij", basis_x, basis_y))
    basis_tensor = torch.stack(basis, dim=0) #  FHW
       
    """ Create the filter by combining the basis with the coefficients, alpha."""
    basis_filter = None 
    if alphas is not None:
        # [out_channels,in_channels,kernel_size[0],kernel_size[1]]
        basis_filter = torch.einsum("fck,fhw->kchw", alphas, basis_tensor)
    return basis_filter, basis_tensor, gauss, hermite


""" Hermite, recursive implementation. It's slow and recursivity does not work well with pytorch.
Inputs:
    - x: the input grid
    - order: the order of the derivative
"""
def hermite_recursive(x, order): # Physicists hermite
    assert(order>=0.0)
    if order==0.0:
        return (x * 0.0 + 1.0)

    elif order==1.0:
        # H{1}(x) = 2 x 
        return 2.0 * x
    
    else:
        # H{n}(x) = 2x H{n-1}(x) - 2(n-1) H{n-2}(x)
        return 2.0*x*hermite_recursive(x, order-1.0) - 2.0*(order-1.0) \
                * hermite_recursive(x, order-2.0)


""" 0-order Hermite polynomial. """
def hermite_0(x):
    return (x*0.0+1.0)


""" 1-order Hermite polynomial. """
def hermite_1(x):
    # H{1}(x) = x 
    return 2.0*x
    

""" 2-order Hermite polynomial. """
def hermite_2(x):
    # H{2}(x) = 4 x^2 - 2
    return (4.0*torch.pow(x,2.0) - 2.0)
 
""" 3-order Hermite polynomial. """
def hermite_3(x):
    # H{3}(x) = 8 x^3 - 12x 
    return (8.0*torch.pow(x,3.0) - 12.0 * x)

""" 4-order Hermite polynomial. """
def hermite_4(x):
    # H{4}(x) = 16 x^4 - 48 x^2 + 12 
    return (16.0*torch.pow(x,4.0) - 48.0*torch.pow(x,2.0) + 12.0)
   
""" 5-order Hermite polynomial. """
def hermite_5(x):
    # H{5}(x) = 32 x^5 - 160 x^3 + 120 x
    return (32.0*torch.pow(x,5.0) - 160.0*torch.pow(x,3.0) + 120.0*x)
    
""" 6-order Hermite polynomial. """
def hermite_6(x):
    # H{6}(x) = 64 x^6 - 480 x^4 + 720 x^2 - 120
    return (64.0*torch.pow(x,6.0) - 480.0*torch.pow(x,4.0) \
            + 720.0*torch.pow(x,2.0) - 120.0)

""" 7-order Hermite polynomial. """
def hermite_7(x):
    # H{7}(x) = 128 x^7 - 1344 x^5 + 3360 x^3 - 1680 x
    return (128.0*torch.pow(x,7.0) - 1344.0*torch.pow(x,5.0) \
            + 3360.0*torch.pow(x,3.0) - 1680.0*x)

""" 8-order Hermite polynomial. """
def hermite_8(x):
    # H{8}(x) = 256 x^8 - 3584 x^6 + 13440 x^4 - 13440 x^2 + 1680
    return (256.0*torch.pow(x,8.0) - 3584.0*torch.pow(x,6.0) \
            + 13440.0*torch.pow(x,4.0) - 13440.0*torch.pow(x,2.0) + 1680.0)

""" 9-order Hermite polynomial. """
def hermite_9(x):
    # H{9}(x) = 512 x^9 - 9216 x^7 + 48384 x^5 - 80640 x^3 + 30240 x
    return (512.0*torch.pow(x,9.0) - 9216.0*torch.pow(x,7.0) \
            + 48384.0*torch.pow(x,5.0) - 80640.0*torch.pow(x,3.0) \
            + 30240.0*x)

""" 10-order Hermite polynomial. """
def hermite_10(x):
    # H{10}(x) = 1024 x^10 - 23040 x^8 - 161280 x^6 - 403200 x^4 \
    #           + 302400 x^2 - 30240
    return (1024.0*torch.pow(x,10.0) - 23040.0*torch.pow(x,8.0) \
            + 161280.0*torch.pow(x,6.0) - 403200.0*torch.pow(x,4.0) \
            + 302400.0*torch.pow(x,2.0) - 30240.0)


""" Switching between the Hermite orders."""
switcher = {
        0: hermite_0,
        1: hermite_1,
        2: hermite_2,
        3: hermite_3,
        4: hermite_4,
        5: hermite_5,
        6: hermite_6,
        7: hermite_7,
        8: hermite_8,
        9: hermite_9,
        10: hermite_10 }


""" Calls the Hermite polynomial computation of a certain order.
Input:
    - x: the grid
    - order: the order of the derivative.
"""
def get_hermite(x, order):
    assert(order>=0.0)
    try:
        func = switcher.get(int(order))
    except:
        func = hermite_recursive
        return func(x,order)
    return func(x)

""" Get the Gaussian basis using Hermite polynomials.
Input:
    - x: the grid
    - order: the order of the derivative.
    - sigma: The sigma of the Gaussian
    - hermite: temporary variable, initialized as the grid.
"""
def get_basis(
        x,
        order,
        gauss,
        sigma,
        hermite):
    # dg^n / dx^n = ( -1/(sqrt(2)sigma) ) ^n H(x / (sqrt(2) sigma)) g

    hermite = get_hermite(torch.div(x, math.sqrt(2.0)*sigma), order) 
    return torch.pow(torch.div(-1.0, math.sqrt(2.0) * sigma), order) \
            * hermite * gauss, hermite

""" Plotting function for plotting the Gaussian derivatives 
(Visually compared to the scipy derivatives).
Input:
    - g_srf: the Gaussian basis
    - sigma: The sigma of the Gaussian
    - order: the order of the derivative.
    - truncate: the spatial extent
    - title: plot title 
"""
def plot2g(g_srf, sigma, order, truncate=2, title=''):        

    filtersize = int(math.ceil(truncate*sigma+0.5))
    x = np.zeros(shape=((2*filtersize+1),(2*filtersize+1)))
    x[filtersize, filtersize] = 1
    g_scipy = []
        
    for i in range(0, int(order)+1):
        for j in range(int(order)-i, -1, -1):
                g_scipy.append(ndimage.filters.gaussian_filter(\
                        x, sigma=sigma, order=(i,j), truncate=truncate))

    import matplotlib.pyplot as plt
    for i in range(0, len(g_scipy)):
        plt.subplot(1, 2, 1)
        plt.title('SRF - '+str(i)+" "+title)
        plt.imshow(g_srf[i].numpy())
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Scipy - '+str(i)+" "+title)
        plt.imshow(g_scipy[i])
        plt.axis('off')
        plt.show()




