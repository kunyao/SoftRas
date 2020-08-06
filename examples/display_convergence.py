import numpy as np
import matplotlib.pyplot as plt
# load files
# file0 = './results/deform_reg_car_shrink_10/2/3d_iou.txt'
# file1 = './results/deform_reg_car_shrink_10/2/2d_iou.txt'
# file2 = './results/deform_noreg_car_shrink_10/2/3d_iou.txt'
# file3 = './results/deform_noreg_car_shrink_10/2/2d_iou.txt'
# file4 = './results/deform_soft_car/2/3d_iou.txt'
# file5 = './results/deform_soft_car/2/2d_iou.txt'
file0 = './results/deform_reg_car_extend_10/2/3d_iou.txt'
file1 = './results/deform_reg_car_extend_10/2/2d_iou.txt'
file2 = './results/deform_noreg_car_extend/2/3d_iou.txt'
file3 = './results/deform_noreg_car_extend/2/2d_iou.txt'
file4 = './results/deform_soft_car_extend/2/3d_iou.txt'
file5 = './results/deform_soft_car_extend/2/2d_iou.txt'

with open(file0, 'r') as f:
    l0 = np.loadtxt(f)
with open(file1, 'r') as f:
    l1 = np.loadtxt(f)
with open(file2, 'r') as f:
    l2 = np.loadtxt(f)
with open(file3, 'r') as f:
    l3 = np.loadtxt(f)
with open(file4, 'r') as f:
    l4 = np.loadtxt(f)
with open(file5, 'r') as f:
    l5 = np.loadtxt(f)

# display iterations
plt.plot(l0[:,0], l0[:,1], label='ss-reg 3d')
plt.plot(l4[:,0], l4[:,1], label='soft 3d')
plt.plot(l2[:,0], l2[:,1], label='local 3d')
plt.plot(l1[:,0], 1 - l1[:,1], label='ss-reg n2d')
plt.plot(l5[:,0], 1 - l5[:,1], label='soft n2d')
plt.plot(l3[:,0], 1 - l3[:,1], label='local n2d')
plt.legend(loc='center right')
plt.xlabel('Number of Iterations')
plt.ylabel('IoU')
plt.grid(True)
# save to image
plt.savefig('out.png')
