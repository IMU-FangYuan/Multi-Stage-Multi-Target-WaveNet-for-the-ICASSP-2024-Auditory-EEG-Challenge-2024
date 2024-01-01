 

import os
import glob
import json
import numpy as np
import glob
import json
import logging
import os, sys
from wavenet.mtnet  import MTWAVE
import sys
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch

import os
import glob
import json
import numpy as np
#from task2_regression.models.linear import simple_linear_model
# from task2_regression.models.vlaai import vlaai, pearson_loss, pearson_metric, pearson_tf_non_averaged


if __name__ == '__main__':

    # Parameters
    window_length_s = 30*64  # 30 seconds
    # Root dataset directory containing test set
    # Parameters
    # Length of the decision window
    fs = 64
 
    gpu = 0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    window_length = window_length_s * fs  # 5 seconds
    # Hop length between two consecutive decision windows
    '''
    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    util_folder = os.path.join(os.path.dirname(task_folder), "util")
    config_path = os.path.join(util_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test
    data_folder = os.path.join(config["dataset_folder"],  config["test_folder"], 'TASK2_regression')
    eeg_folder = os.path.join(data_folder, 'preprocessed_eeg')
    '''
    data_folder = os.path.join('/ddnstor/auditorycode/test_data_task2/')
    eeg_folder = os.path.join(data_folder, 'preprocessed_eeg')

    # uncomment if you want to train with the mel spectrogram stimulus representation
    stimulus_features = ["mel"]
    stimulus_dimension = 10

    features = ["eeg"] + stimulus_features

    #pretrained_model = os.path.join(os.path.dirname(__file__), 'results_linear_baseline', 'model.h5')

    # Define and load the pretrained model
   
    model = MTWAVE( in_channels=64, res_channels=128, skip_channels=128, out_channels=10, num_res_layers=36, dilation_cycle=12).to(device)
    ckpt = torch.load('avgwav1noval.ckpt',map_location=torch.device('cpu') )
    
    model.load_state_dict(ckpt )
    
    model.eval()
    

    test_eeg_mapping = glob.glob(os.path.join(data_folder, 'sub*mapping.json'))

    for sub_stimulus_mapping in test_eeg_mapping:
        subject = os.path.basename(sub_stimulus_mapping).split('_')[0]

        # load stimulus mapping
        sub_stimulus_mapping = json.load(open(sub_stimulus_mapping))

        #load eeg data
        sub_path = os.path.join(eeg_folder, f'{subject}_eeg.npz')
        sub_eeg_data = dict(np.load(sub_path))

        data_eeg =  np.stack([sub_eeg_data[value['eeg']]  for key, value in sub_stimulus_mapping.items() ])

        id_list= list(sub_stimulus_mapping.keys())

        # predict
        data_eeg = torch.from_numpy(data_eeg).to(device).float()
        
       
         
        #inputs, envs,labels = mixup2eeg_and_mel(inputs,envs, labels)
        data_mean =data_eeg.mean(dim=1).unsqueeze(1)
        data_std = data_eeg.std(dim=1).unsqueeze(1)
        data_eeg = (data_eeg - data_mean) / data_std
        print(data_eeg.shape)
        with torch.no_grad():
            predictions = model(data_eeg)[1]
        print(predictions.shape)
         
        
        predictions = predictions.cpu().numpy()
        # Make predictions json-serializable
        predictions = [np.array(value).tolist() for value in np.squeeze(predictions)]

        # Create dictionary from id_list and predictions
        sub = dict(zip(id_list, predictions))

        prediction_dir = os.path.join(os.path.dirname(__file__), 'predictionswav1noval')
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, subject + '.json'), 'w') as f:
            json.dump(sub, f)


