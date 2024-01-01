# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
import os
import argparse
import glob

import yaml
import numpy as np
import torch
 

def average():
    checkpoints = []

    dir_path = '/ddnstor/auditorycode/test_results/wav1/'
    path_list = os.listdir(dir_path)
    print(path_list)
 
 
    avg = None
    num = len(path_list)
    
    for path in path_list:
        path = os.path.join(dir_path, path)
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))['state_dict']
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                 
                avg[k] += states[k]
                
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
     
    torch.save(avg, 'avgwav1new.ckpt')


def main():
 
    average()


if __name__ == '__main__':
    main()
