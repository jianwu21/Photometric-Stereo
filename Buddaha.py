# coding: utf-8
from processing import *
from numpy import *
import matplotlib.pyplot as plt

I,mask,S = read_data_file("Data/Buddha.mat")

plt.figure(figsize=(8,5))
for i in range(I.shape[2]):
    plt.subplot(2,I.shape[2]/2,i+1)
    plt.imshow(I[:,:,i],cmap= plt.cm.gray)
    plt.axis('off')
    plt.title(i)

I_all = I.reshape(I.shape[0]*I.shape[1],10).T

M = linalg.pinv(S).dot(I_all)
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

plt.imshow()

