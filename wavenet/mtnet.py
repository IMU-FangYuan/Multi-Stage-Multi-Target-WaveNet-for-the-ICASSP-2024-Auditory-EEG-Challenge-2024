import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from numpy.lib import stride_tricks
from wavenet.waveneteeg2env import WaveNet_regressor 
from wavenet.models import MultiLayerCrossAttention
from torch.autograd import Variable

 
class MTWAVE(nn.Module):
 
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers, dilation_cycle):
        super(MTWAVE, self).__init__()
      
        self.eeg2envblock = WaveNet_regressor( in_channels=in_channels, res_channels=res_channels, skip_channels=skip_channels, out_channels=1, num_res_layers=num_res_layers, dilation_cycle=dilation_cycle)        
        self.block1 =  wave_cross_block( in_channels=in_channels, res_channels=res_channels, skip_channels=skip_channels, out_channels=out_channels, num_res_layers=num_res_layers, dilation_cycle=dilation_cycle,prechannel = 1)
        self.block2 =  wave_cross_block( in_channels=in_channels, res_channels=res_channels, skip_channels=skip_channels, out_channels=80, num_res_layers=num_res_layers, dilation_cycle=dilation_cycle,prechannel=10)
        self.block3 =  wave_cross_block( in_channels=in_channels, res_channels=512, skip_channels=512, out_channels=257, num_res_layers=num_res_layers, dilation_cycle=dilation_cycle,prechannel=80)
        
    def forward(self, eeg):
 
         
        env = self.eeg2envblock(eeg) #[b t f]
         
        mel = self.block1(env,eeg)  
        mel80 = self.block2(mel,eeg)
        
        spec = self.block3(mel80,eeg)
        return env,mel ,mel80,spec


class wave_cross_block(nn.Module):
     
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers, dilation_cycle,prechannel):
        super(wave_cross_block, self).__init__()
        self.Fast1 = WaveNet_regressor( in_channels=in_channels, res_channels=res_channels, skip_channels=skip_channels, out_channels=out_channels, num_res_layers=num_res_layers, dilation_cycle=dilation_cycle)
        self.crossattn1 =  MultiLayerCrossAttention(input_size=1920, layer=2, in_ch=64, kernel_size=3, dilation=1)
        self.proj1 = nn.Linear(prechannel,64)
        self.crossattn2 =  MultiLayerCrossAttention(input_size=1920, layer=2, in_ch=64, kernel_size=3, dilation=1)
        self.crossattn3 =  MultiLayerCrossAttention(input_size=1920, layer=2, in_ch=64, kernel_size=3, dilation=1)
          
    def forward(self, pre_out,eeg):
        
        #out1 = self.crossattn1(self.proj1(pre_out).transpose(-1,-2),eeg.transpose(-1,-2)) #inpt [b 640 64] 
        
        pre_out = self.proj1(pre_out)
        pre_out = self.crossattn2(pre_out.transpose(-1,-2),pre_out.transpose(-1,-2))
        eeg = self.crossattn3(eeg.transpose(-1,-2),eeg.transpose(-1,-2))
        out1 = self.crossattn1(pre_out ,eeg )  
         
        out1 = self.Fast1(out1.transpose(-1,-2))
        
        return out1 
     
        
if __name__ == '__main__':
 
    net =  MTWAVE( in_channels=64, res_channels=128, skip_channels=128, out_channels=10, num_res_layers=36, dilation_cycle=12).cuda()
    indata = torch.randn((2,1920,64), dtype=torch.float32).cuda()
     
    out = net(indata)
    print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)
    