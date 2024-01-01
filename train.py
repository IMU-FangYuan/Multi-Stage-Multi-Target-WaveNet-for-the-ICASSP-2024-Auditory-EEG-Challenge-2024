import glob
import os
import torch
import torch.nn as nn
import torchaudio
import argparse
import numpy as np
from util.utils import  save_checkpoint 
from torch.optim.lr_scheduler import StepLR
from wavenet.mtnet  import MTWAVE
from util.cal_pearson import l1_loss, pearson_loss, pearson_metric
from util.datasetwav import RegressionDataset
import logging as log
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch.nn.functional import pairwise_distance
 

log.basicConfig(filename='trainwav1.log', level=log.INFO)
log.info('This is an info message')
parser = argparse.ArgumentParser()

parser.add_argument('--epoch',type=int, default=3000)
parser.add_argument('--batch_size',type=int, default=24)
parser.add_argument('--win_len',type=int, default = 30)
parser.add_argument('--sample_rate',type=int, default = 64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--g_con', default=True, help="experiment for within subject")
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--in_channel', type=int, default=64, help="channel of the input eeg signal") 
parser.add_argument('--lamda',type=float,default=0.2)
parser.add_argument('--writing_interval', type=int, default=40)
parser.add_argument('--saving_interval', type=int, default=40)
parser.add_argument('--dataset_folder',type= str, default="/ddnstor/project/auditoryeegchallenge/auditoryeegdata/", help='write down your absolute path of dataset folder')
parser.add_argument('--experiment_folder',default='wav1', help='write down experiment name')

args = parser.parse_args()

 # Set the arameters and device.

input_length = args.sample_rate * args.win_len 
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(device)
# Provide the path of the dataset.

data_folder = os.path.join(args.dataset_folder)
features =  ["eeg"] + ["mel"] + ["envelope"] +["wav"]

# Create a directory to store (intermediate) results.
result_folder = 'test_results'
experiment_folder = args.experiment_folder
save_path = os.path.join(result_folder, experiment_folder)
 
if not os.path.exists(save_path):
    os.makedirs(save_path)
    

def mixup2eeg_and_mel(inputs,envs, labels,wav):
    batch_size, time_steps, num_features = inputs.size()
    half_batch_size = batch_size // 2
    mixup_ratio = torch.rand(half_batch_size, device=inputs.device).view(-1, 1, 1)
    inputs_part1, inputs_part2 = torch.split(inputs , [half_batch_size, half_batch_size])
    envs_part1, envs_part2 = torch.split(envs, [half_batch_size, half_batch_size])
    labels_part1, labels_part2 = torch.split(labels, [half_batch_size, half_batch_size])
    wav_part1, wav_part2 = torch.split(wav, [half_batch_size, half_batch_size])
    mixed_inputs = mixup_ratio * inputs_part1 + (1 - mixup_ratio) * inputs_part2
    mixed_envs = mixup_ratio * envs_part1 + (1 - mixup_ratio) * envs_part2
    mixed_labels = mixup_ratio * labels_part1 + (1 - mixup_ratio) * labels_part2
    mixed_wav = mixup_ratio.view(-1, 1 ) * wav_part1 + (1 - mixup_ratio).view(-1, 1 ) * wav_part2
   

    return mixed_inputs,mixed_envs, mixed_labels,mixed_wav
    
    
def main():

    # Set the model and optimizer, scheduler.
         
    best_val_metric = 0
    #torch.manual_seed(3407)
    
    model = MTWAVE( in_channels=64, res_channels=128, skip_channels=128, out_channels=10, num_res_layers=36, dilation_cycle=12).to(device)
   
 
    
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate,
                                betas=(0.9, 0.98),
                                eps=1e-09)
   
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
     
    # Define train set and loader.
    tr_files = []
    tt_files = []
    v_files = []
    train_files = [x for x in glob.glob(os.path.join(data_folder, "*-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    for fpath in train_files:
        sub_idx = int(fpath.split('/')[-1].split('_-_')[1].split('-')[-1])
        if sub_idx>=27 and  sub_idx<=42:
            v_files.append(fpath)
        else:
            tr_files.append(fpath)
             
    train_set= RegressionDataset(tr_files, input_length, args.in_channel, 'train', args.g_con)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size = args.batch_size,
            num_workers = 2,
            sampler = None,
            drop_last=True,
            shuffle=True)

    # Define validation set and loader.
    
    val_set = RegressionDataset(v_files, input_length, args.in_channel, 'val', args.g_con)
    val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size = 1,
            num_workers = 2,
            sampler = None,
            drop_last=True,
            shuffle=False)

    # Define test set and loader.

    kl_loss = nn.KLDivLoss(reduction="mean", log_target=True)
    # Train the model.
    spec_transform = torchaudio.transforms.Spectrogram(n_fft = 512, win_length  = 400, hop_length = 250, center = False, return_complex  = None).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,  
        win_length = 400,
        hop_length=250,   
        n_mels=80,   
        center=False, norm='slaney'
      ).to(device)
      
    for epoch in range(args.epoch):
        model.train()
        train_loss = train_outp = 0
        l_1 =l_p= l_k= 0
         
        for inputs, labels,envs,wav,_ in train_dataloader:
            optimizer.zero_grad()
            
            inputs = inputs.to(device) 
            labels = labels.to(device)
            sub_id = sub_id.to(device)
            envs = envs.to(device)
            wav = wav.to(device)
            
            #mixup all data [4b t c]->[b t c]
            inputs, envs,labels,wav = mixup2eeg_and_mel(inputs,envs, labels,wav)
            inputs, envs,labels,wav = mixup2eeg_and_mel(inputs,envs, labels,wav)
            
            #norm 
            data_mean =inputs.mean(dim=1).unsqueeze(1)
            data_std = inputs.std(dim=1).unsqueeze(1)
            inputs = (inputs - data_mean) / data_std
            
            env_mean =envs.mean(dim=1).unsqueeze(1)
            env_std = envs.std(dim=1).unsqueeze(1)
            envs = (envs - env_mean) / env_std
            
              
            labels_mean = labels.mean(dim=1).unsqueeze(1)
            labels_std = labels.std(dim=1).unsqueeze(1)
            labels = (labels - labels_mean) / labels_std
         
                       
            wav_mean = wav.mean(dim=1).unsqueeze(1)
            wav_std = wav.std(dim=1).unsqueeze(1)
            wav = (wav - wav_mean) / wav_std
            
            #gen mel80 and mag using 16k wav data
            mel80 =  mel_transform(F.pad(wav, (250, 250), value=0)).transpose(-1,-2)
            mag = spec_transform(F.pad(wav, (250, 250), value=0)).transpose(-1,-2)
            
            
            pred_env,pred_mel,pred_mel80,pred_mag = model(inputs) 
            
            #loss 0.2l1+lp+l_kl
            l_p2 =  pearson_loss(pred_mel, labels) 
            l_p = pearson_loss(pred_env, envs).mean() + l_p2.mean() + pearson_loss(pred_mel80, mel80).mean()  + pearson_loss(pred_mag, mag).mean() 
            l_1 = l1_loss(pred_env, envs).mean() + l1_loss(pred_mel, labels).mean() + l1_loss(pred_mel80, mel80).mean() + l1_loss(pred_mag, mag).mean() 
            l_k = kl_loss(F.log_softmax(pred_mel, dim=-1).reshape(inputs.shape[0],-1), F.log_softmax(labels, dim=-1).reshape(inputs.shape[0],-1)).mean()
             
            loss = l_p + args.lamda * l_1+ l_k
            loss = loss.mean()
            loss.backward()
           
            optimizer.step()
            train_loss += loss.item()
            train_outp += l_p2.mean().item()
            
        train_loss /= len(train_dataloader)
        train_outp /= len(train_dataloader)
        if epoch % args.writing_interval == 0:
            print(f'|-Train-|{epoch}: {train_loss:.3f}')
            log.info('success!')
            log.info('train: Epoch [%d/%d], (  l_p : %.4f |  l_1 : %.4f)' % (
                        epoch + 1, 100,  train_outp, train_loss))
         
        # Validate the model.
        val_loss = 0
        val_metric = 0
        val_count = 0.
        if epoch % args.writing_interval == 0:

            model.eval()

            with torch.no_grad():
                for val_inputs, val_labels, _,_,_ in val_dataloader:
                    val_inputs = val_inputs.squeeze(0).to(device) 
                    val_labels = val_labels.squeeze(0).to(device)
                     
                    
                    data_mean =val_inputs.mean(dim=1).unsqueeze(1)
                    data_std = val_inputs.std(dim=1).unsqueeze(1)
                    val_inputs = (val_inputs - data_mean) / data_std
                    
                    mb=val_inputs.shape[0] %4
                    val_inputs=val_inputs[:-mb,:,:]
                    val_inputs=val_inputs.reshape(-1,4,1920,64)
                    val_labels=val_labels[:-mb,:,:]
                    val_labels=val_labels.reshape(-1,4,1920,10)
  
                    for i in range(val_inputs.shape[0]):
                        val_outputs = model(val_inputs[i])[1] 
                        val_count+= 4
                        val_loss   += pearson_loss(val_outputs, val_labels[i]).mean()*4
                        val_metric += pearson_metric(val_outputs, val_labels[i]).mean()*4

                val_loss /= val_count
                val_metric /= val_count
                val_metric = val_metric.mean()
                learning_rate = print(optimizer.param_groups[0]["lr"])
     
                save_checkpoint(model, optimizer, learning_rate, epoch, save_path)    
                print(f'|-Validation-|{epoch}: {val_loss.mean().item():.3f} {val_metric.item():.3f}')
                print("Loss", "Validation",  val_loss, epoch)
                print("Pearson", "Validation",  val_metric, epoch)
                log.info('val : Epoch [%d/%d], (  val_loss : %.4f |  val_metric : %.4f)' % (
                        epoch + 1, 100,  val_loss, val_metric))
               

        scheduler.step()


if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING=1
    main()
