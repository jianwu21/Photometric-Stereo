from numpy import *
from ps_utils import *
import matplotlib.pyplot as plt 

# read the .mat file
I,mask,S = read_data_file("Beethoven.mat")

for i in range(I.shape[2]):
    plt.subplot(1,3,i+1)
    plt.imshow(I[:,:,i],cmap = plt.cm.gray)
    plt.axis('off')

I_all = vstack((I[:,:,0].flatten(),I[:,:,1].flatten(),I[:,:,2].flatten()))

M = linalg.inv(S).dot(I_all)
M_1 = M[0]
M_2 = M[1]
M_3 = M[2]


plt.figure()
rho = sqrt(M_1**2 + M_2**2 + M_3**2).reshape(256,256)
plt.imshow(rho,cmap = plt.cm.gray)
plt.title("2-D Image for albedo")


n_1 = (M_1 / rho.reshape(-1)).reshape(256,256)
n_2 = (M_2 / rho.reshape(-1)).reshape(256,256)
n_3 = (M_3 / rho.reshape(-1)).reshape(256,256)

z = unbiased_integrate(n_1,n_2,n_3,mask)

display_depth_matplotlib(z)

display_depth_mayavi(z)

