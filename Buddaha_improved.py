
# coding: utf-8
from ps_utils import *
from numpy import *
import matplotlib.pyplot as plt


I,mask,S = read_data_file("Buddha.mat")

I_all = I.reshape(I.shape[0]*I.shape[1],10).T

from itertools import *

all_iter = [(i,j,k) for i in [0,1,2,3] for j in [4,5] for k in [6,7,8,9]]

num = 1
plt.figure(figsize=(8,8))
for i,j,k in all_iter:
    I_choosen = vstack((I_all[i],I_all[j],I_all[k]))
    S_choosen = vstack((S[i],S[j],S[k]))
    
    M = linalg.pinv(S_choosen).dot(I_choosen)
    M_1 = M[0]
    M_2 = M[1]
    M_3 = M[2]
    
    rho = sqrt(M_1**2 + M_2**2 + M_3**2).reshape(I.shape[0],I.shape[1])

    n_1 = (M_1 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1])
    n_2 = (M_2 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1])
    n_3 = (M_3 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1])
    
    z = unbiased_integrate(n_1,n_2,n_3,mask)
    
    plt.subplot(8,4,num)
    plt.imshow(z*mask)
    plt.axis('off')
    plt.title(num)
    num += 1


# choose the all_iter[9]
i,j,k = all_iter[9]
I_choosen = vstack((I_all[i],I_all[j],I_all[k]))
S_choosen = vstack((S[i],S[j],S[k]))


M = linalg.inv(S_choosen).dot(I_choosen)
M_1 = M[0]
M_2 = M[1]
M_3 = M[2]


plt.figure()
rho = sqrt(M_1**2 + M_2**2 + M_3**2).reshape(I.shape[0],I.shape[1])
plt.imshow(rho,cmap = plt.cm.gray)
plt.title("2-D Image for albedo")

n_1 = (M_1 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1])
n_2 = (M_2 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1])
n_3 = (M_3 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1])

z = unbiased_integrate(n_1,n_2,n_3,mask)

display_depth_matplotlib(z)
display_depth_mayavi(z)

plt.show()
