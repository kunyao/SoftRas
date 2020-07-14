import numpy as np
import matplotlib.pyplot as plt
# load files
file0 = './results/deform_car_soft_extend/3d_iou.txt'
file1 = './results/deform_car_reg20_1.0_10_extend/3d_iou.txt'
file2 = './results/deform_car_noreg_extend/5/3d_iou.txt'

with open(file0, 'r') as f:
    l0 = np.loadtxt(f)
with open(file1, 'r') as f:
    l1 = np.loadtxt(f)
with open(file2, 'r') as f:
    l2 = np.loadtxt(f)

# display iterations
plt.plot(l0[:,0], l0[:,1], label='soft')
plt.plot(l1[:,0], l1[:,1], label='s-reg')
plt.plot(l2[:,0], l2[:,1], label='vanilla')
plt.legend(loc='upper right')
# save to image
plt.savefig('out.png')
