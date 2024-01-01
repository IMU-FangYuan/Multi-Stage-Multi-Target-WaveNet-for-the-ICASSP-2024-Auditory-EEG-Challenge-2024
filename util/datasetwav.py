import torch
import itertools
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import pdb
from random import randint

class RegressionDataset(Dataset):
    """Generate data for the regression task."""

    def __init__(
        self,
        files,
        input_length,
        channels,
        task,
        g_con = True
    ):

        self.input_length = input_length
        self.files = self.group_recordings(files)
        self.channels = channels
        self.task = task
        self.g_con = g_con

    def group_recordings(self, files):
 
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
    
        return new_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, recording_index):
        
        # 1. For within subject, return eeg, envelope and subject ID
        # 2. For held-out subject, return eeg, envelope

        if self.task == "train":
            x, env, y, wav, sub_id = self.__train_data__(recording_index)

        else:
            x, env, y, wav, sub_id = self.__test_data__(recording_index)

        return x, y, env, wav, sub_id


    def __train_data__(self, recording_index):

        framed_data = []

        for idx, feature in enumerate(self.files[recording_index]):
            data = np.load(feature)

            if idx == 0: 
                start_idx= randint(0,len(data)- self.input_length)
             
            if idx == 3:  #wav file
                framed_data += [data[start_idx*250:start_idx*250 + 250*self.input_length]]
            else:
                framed_data += [data[start_idx:start_idx + self.input_length]]

        if self.g_con == True:
            sub_idx = feature.split('/')[-1].split('_-_')[1].split('-')[-1]
            sub_idx = int(sub_idx) - 1 
        
        else:
            sub_idx = torch.FloatTensor([0])
    
            # return torch.FloatTensor(framed_data[0]), torch.FloatTensor(framed_data[1]), sub_idx
        
            
        return torch.FloatTensor(framed_data[0]), torch.FloatTensor(framed_data[1]), torch.FloatTensor(framed_data[2]), torch.FloatTensor(framed_data[3]),sub_idx

    def __test_data__(self, recording_index):
        """
        return: list of segments [[eeg, envelope] ...] depending on self.input_length 
                e.g.,for 10 second-long input signal and input_length==5, return [[5, 5], [5, 5]]
        
        """
        framed_data = []

        for idx, feature in enumerate(self.files[recording_index]):
            data = np.load(feature)
            
            if idx == 3:  #wav file
                nsegment = data.shape[0] // (self.input_length * 250)
                data = data[:int(nsegment * self.input_length*250)]
                segment_data = [torch.FloatTensor(data[i:i+self.input_length*250]).unsqueeze(0) for i in range(0, data.shape[0], self.input_length*250)]
                segment_data = torch.cat(segment_data)
                framed_data += [segment_data]
            else:
                nsegment = data.shape[0] // self.input_length
                data = data[:int(nsegment * self.input_length)]
                segment_data = [torch.FloatTensor(data[i:i+self.input_length]).unsqueeze(0) for i in range(0, data.shape[0], self.input_length)]
                segment_data = torch.cat(segment_data)
                framed_data += [segment_data]
            
        if self.g_con == True:
            sub_idx = feature.split('/')[-1].split('_-_')[1].split('-')[-1]
            sub_idx = int(sub_idx) - 1    

        else:
            sub_idx = torch.FloatTensor([0])

        return framed_data[0], framed_data[1],framed_data[2],framed_data[3] ,sub_idx
        
        
        
        
if __name__ == '__main__':
        
    # Define train set and loader.
    data_folder = '/ddnstor/imu_liuxin/fyuan/auditoryeegdatanew/'
    features = ["eeg"] + ["mel"] + ["envelope"] +["wav"]
    train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    train_set= RegressionDataset(train_files, 640, 64, 'test', True)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size = 1,
            num_workers = 1,
            sampler = None,
            drop_last=True,
            shuffle=True)
    for inputs, labels,envs,wav,sub_id in train_dataloader:
        
        print(inputs.shape, "eeg.shape \n")
        print(labels.shape, "mel.shape \n")
        print(envs.shape,"env.shape \n")
        print(wav.shape,"wav.shape \n")
        
        print(sub_id)
       
