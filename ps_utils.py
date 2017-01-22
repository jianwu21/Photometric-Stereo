# -*- coding: utf-8 -*-
"""
ps_utils.py
------------

Two methods of integration of a normal field to a depth function
by solving a Poisson equation. 
- The first, unbiased, implements a Poisson solver on an irregular domain.
  rather standard approach.
- The second implements the Simchony et al. method for integration of a normal
  field.
 
They are port of Yvain Quéau's Matlab implementation to Python.
See Yvain's code for more!

Added to them, a function to read a dataset from a Matlab mat-file
and a function to display a depth surface using Mayavi. 

Author: Francois Lauze, University of Copenhagen
Date December 2015 / January 2016
"""

import numpy as np
from scipy import fftpack as fft
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
from matplotlib import pyplot as plt

def cdx(f):
    """
    central differences for f-
    """
    m = f.shape[0]
    west = [0] + range(m-1)
    east = range(1,m) + [m-1]
    return 0.5*(f[east,:] - f[west,:])
    
def cdy(f):
    """
    central differences for f-
    """
    n = f.shape[1]
    south = [0] + range(n-1)
    north = range(1,n) + [n-1]
    return 0.5*(f[:,north] - f[:,south])
    
def sub2ind(shape, X, Y):
    """
    An equivalent of Matlab sub2ind, but without 
    argument checking and for dim 2 only.
    """    
    Z = np.array(zip(X,Y)).T
    shape = np.array(shape)
    indices = np.dot(shape, Z)
    indices.shape = indices.size
    return indices
    
def tolist(A):
    """
    Linearize array to a 1D list
    """
    return list(np.reshape(A, A.size))
    

    
def simchony_integrate(n1, n2, n3, mask):
    """
    Integration of the normal field recovered from observations onto 
    a depth map via Simchony et al. hybrid DCT / finite difference
    methods.
    
    Done by solving via DCT a finite difference equation discretizing
    the equation:
        Laplacian(z) - Divergence((n1/n3, n2/n3)) = 0
    under proper boundary conditions ("natural" boundary conditions on 
    a rectangular domain)
    
    Arguments:
    ----------
    n1, n2, n3: nympy float arrays 
        the 3 components of the normal field. They must be 2D arrays
        of size (m,n). Array (function) n3 should never be 0.
       
    Returns:
    --------
        z : depth map obtained by integration of the field -n1/n3, -n2/n3
        Set to nan outside the mask.
    """
    # first a bit paranoid, so check arguments
    if (type(n1) != np.ndarray) or (type(n2) != np.ndarray) or (type(n3) != np.ndarray):
        raise TypeError('One or more arguments are not numpy arrays.')
        
    if (len(n1.shape) != 2) or (len(n2.shape) != 2) or (len(n3.shape) != 2):
        raise TypeError('One or more arguments are not 2D arrays.')

    if (n1.shape != n2.shape) or (n1.shape != n3.shape):
        raise TypeError('Array dimensions mismatch.')

    try:
        n1 = n1.astype(float)
        n2 = n2.astype(float)
        n3 = n3.astype(float)
    except:
        raise TypeError('Arguments not all (convertible to) float.')
        
        
    # Hopefully on the safe side now
    m,n = n1.shape
        
    p = -n1/n3
    q = -n2/n3

    # divergence of (p,q)
    px = cdx(p)
    qy = cdy(q)
    
    f = px + qy      

    # 4 edges
    f[0,1:-1]  = 0.5*(p[0,1:-1] + p[1,1:-1])    
    f[-1,1:-1] = 0.5*(-p[-1,1:-1] - p[-2,1:-1])
    f[1:-1,0]  = 0.5*(q[1:-1,0] + q[1:-1,1])
    f[1:-1,-1] = 0.5*(-q[1:-1,-1] - q[1:-1,-2])

    # 4 corners
    f[ 0, 0] = 0.5*(p[0,0] + p[1,0] + q[0,0] + q[0,1])
    f[-1, 0] = 0.5*(-p[-1,0] - p[-2,0] + q[-1,0] + q[-1,1])
    f[ 0,-1] = 0.5*(p[0,-1] + p[1,-1] - q[0,-1] - q[1,-1])
    f[-1,-1] = 0.5*(-p[-1,-1] - p[-2,-1] -q[-1,-1] -q[-1,-2])

    # cosine transform f (reflective conditions, a la matlab, 
    # might need some check)
    fs = fft.dct(f, axis=0, norm='ortho')
    fs = fft.dct(fs, axis=1, norm='ortho')

    # check that this one works in a safer way than Matlab!
    x, y = np.mgrid[0:m,0:n]
    denum = (2*np.cos(np.pi*x/m) - 2) + (2*np.cos(np.pi*y/n) -2)
    Z = fs/denum
    Z[0,0] = 0.0 
    # or what Yvain proposed, it does not really matters
    # Z[0,0] = Z[1,0] + Z[0,1]
    
    z = fft.dct(Z, type=3, norm='ortho', axis=0)
    z = fft.dct(z, type=3, norm='ortho', axis=1)
    out = np.where(mask == 0)
    z[out] = np.nan
    return z
# simchony()





def unbiased_integrate(n1, n2, n3, mask, order=2):
    """
    Constructs the finite difference matrix, domain and other information
    for solving the Poisson system and solve it. Port of Yvain's implementation, 
    even  respecting the comments :-)
    
    Arguments:
    ----------
    n1, n2, n3, mask: numpy arrays
        all arrays must have the same size, say (m,n)
        n1, n2 and n3 are the three components of field of 
        normal vectors to the surface we want to reconstruct, obtained by 
        solving the system formed by the light sources and measured intensities,
        and then after albedo normalization.
        
    order: int
        default value of 2 is a good idea, don't change it
    
    Returns:
    -------
    z: numpy array
        array of size (m,n), with computed depth values inside the 
        make region (mask > 0) and NaN (Not a Number) in the region mask == 0.
        
        
    It solves for the system comming from the discretization of 
    
           -Laplacian(z) - Divergence(n1/n3, n2/n3) = 0
    
    with some boundary conditions, it gives a discretized Poisson system
        AZ = b
    and z is obtained by mapping the values in Z to the region of the 2D image 
    of size (m,n) where mask > 0. The rest of the z-image values are set to NaN.
    """
    
    p = -n1/n3
    q = -n2/n3        
    
    # Calculate some usefuk masks
    m,n = mask.shape
    Omega = np.zeros((m,n,4))
    Omega_padded = np.pad(mask, (1,1), mode='constant', constant_values=0)
    Omega[:,:,0] = Omega_padded[2:,1:-1]*mask
    Omega[:,:,1] = Omega_padded[:-2,1:-1]*mask
    Omega[:,:,2] = Omega_padded[1:-1,2:]*mask
    Omega[:,:,3] = Omega_padded[1:-1,:-2]*mask
    del Omega_padded
    
    # Mapping    
    indices_mask = np.where(mask > 0)
    lidx = len(indices_mask[0])
    mapping_matrix = np.zeros(p.shape, dtype=int)
    mapping_matrix[indices_mask] = xrange(lidx)
    
    if order == 1:
        pbar = p.copy()
        qbar = q.copy()
    elif order == 2:
        pbar = 0.5*(p + p[range(1,m) + [m-1], :])
        qbar = 0.5*(q + q[:, range(1,n) + [n-1]])
        
    # System
    I = []
    J = []
    K = []
    b = np.zeros(lidx)


    # In mask, right neighbor in mask
    rset = Omega[:,:,2]
    X, Y = np.where(rset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X, Y+1)
    I_neighbors = mapping_matrix[(X,Y+1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= qbar[(X,Y)]
    
	
    #	In mask, left neighbor in mask
    lset = Omega[:,:,3]
    X, Y = np.where(lset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X, Y-1)
    I_neighbors = mapping_matrix[(X,Y-1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)  
    b[I_center] += qbar[(X,Y-1)]


    # In mask, top neighbor in mask
    tset = Omega[:,:,1]
    X, Y = np.where(tset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X-1, Y)
    I_neighbors = mapping_matrix[(X-1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] += pbar[(X-1,Y)]


    #	In mask, bottom neighbor in mask
    bset = Omega[:,:,0]
    X, Y = np.where(bset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X+1, Y)
    I_neighbors = mapping_matrix[(X+1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= pbar[(X,Y)]
    
    # Construction de A : 
    A = sp.csc_matrix((K, (I, J)))
    A = A + sp.eye(A.shape[0])*1e-9
    z = np.nan*np.ones(mask.shape)
    z[indices_mask] = spsolve(A, b)
    return z
    


def display_depth_mayavi(z):
    """
    Display the computed depth function as a surface using 
    mayavi mlab.
    """
    from mayavi import mlab
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    
    surf = mlab.mesh(x, y, z, colormap="gray")
    mlab.view(azimuth=0, elevation=90)
    surf.actor.property.interpolation = 'phong'
    surf.actor.property.specular = 0.1
    surf.actor.property.specular_power = 5
    mlab.show()
    
    
def display_depth_matplotlib(z):
    """
    Same as above but using matplotlib instead.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LightSource
    
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ls = LightSource(azdeg=0, altdeg=65)
    greyvals = ls.shade(z, plt.cm.Greys)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, facecolors=greyvals)
    plt.axis('off')
    plt.axis('equal')
    plt.show()



    
def display_image(u):
    """
    Display a 2D imag
    """
    plt.imshow(u)
    plt.show()
    
    
    
def read_data_file(filename):
    """
    Read a matlab PS data file and returns
    - the images as a 3D array of size (m,n,nb_images)
    - the mask as a 2D array of size (m,n) with 
      mask > 0 meaning inside the mask
    - the light matrix S as a (nb_images, 3) matrix
    """
    data = loadmat(filename)
    I = data['I']
    mask = data['mask']
    S = data['S']
    return I, mask, S

