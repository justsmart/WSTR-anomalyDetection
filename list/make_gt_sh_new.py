import os
import numpy as np


root_path = "/home/lcl/PycharmProjects/AD/data/mine_sh_test_c3d"

rgb_list_file ='../list/sh-i3d-test.list'
temporal_root = '/home/lcl/ShanghaiTechDataset/Testing/test_frame_mask/'
gt = []
num=0
f=open(rgb_list_file,'r')
npy_paths=f.readlines()
for i,npy in enumerate(npy_paths):
    npy = npy.replace('\n','')
    feat_len=np.load(npy).shape[0]*16
    if i<=43:

        mask_name = npy.split('/')[-1].replace('_i3d.npy','.npy')
        label = np.load(os.path.join(temporal_root,mask_name))
        this_label = label[:feat_len]
        gt.extend(this_label)
        num+=feat_len
    else:
        temp = [0]*feat_len
        gt.extend(temp)
        num += feat_len

np.array(gt,dtype=float)
np.save('gt-sh-new.npy', gt)
print(len(gt),num)