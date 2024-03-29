import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F

 

def swish(x):
    return x * torch.sigmoid(x)


# dilated conv layer with kaiming_normal initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


# conv1x1 layer with zero initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py but the scale parameter is removed
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


# every residual block (named residual layer in paper)
# contains one noncausal dilated conv
class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        # the layer-specific fc for diffusion step embedding
        # self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)

        # add mel spectrogram upsampler and conditioner conv1x1 layer
        #self.upsample_conv2d = torch.nn.ModuleList()
        #for s in [16, 16]:
        #    conv_trans2d = torch.nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
        #    conv_trans2d = torch.nn.utils.weight_norm(conv_trans2d)
        #    torch.nn.init.kaiming_normal_(conv_trans2d.weight)
        #    self.upsample_conv2d.append(conv_trans2d)
        #self.mel_conv = Conv(80, 2 * self.res_channels, kernel_size=1)  # 80 is mel bands

        # residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):

        x = input_data
        h = x
        B, C, L = x.shape
        #print("B",B)
        #print("C",C)
        #print("L",L)
        assert C == self.res_channels

        # add in diffusion step embedding
        #part_t = self.fc_t(diffusion_step_embed)
        #part_t = part_t.view([B, self.res_channels, 1])
        #h = h + part_t

        # dilated conv layer
        h = self.dilated_conv_layer(h)

        # add mel spectrogram as (local) conditioner
        #assert mel_spec is not None

        # Upsample spectrogram to size of audio
        #mel_spec = torch.unsqueeze(mel_spec, dim=1)
        #mel_spec = F.leaky_relu(self.upsample_conv2d[0](mel_spec), 0.4)
        #mel_spec = F.leaky_relu(self.upsample_conv2d[1](mel_spec), 0.4)
        #mel_spec = torch.squeeze(mel_spec, dim=1)

        #assert(mel_spec.size(2) >= L)
       # if mel_spec.size(2) > L:
        #    mel_spec = mel_spec[:, :, :L]

        #mel_spec = self.mel_conv(mel_spec)
       # h += mel_spec

        # gated-tanh nonlinearity
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        # residual and skip outputs
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers

        # the shared two fc layers for diffusion step embedding
        # self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        # self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        # stack all residual blocks with dilations 1, 2, ... , 512, ... , 1, 2, ..., 512
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       dilation=2 ** (n % dilation_cycle)))

    def forward(self, input_data):
        x = input_data

 
        h = x
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n](h)  # use the output from last residual layer
            skip = skip + skip_n  # accumulate all skip outputs

        return skip * math.sqrt(1.0 / self.num_res_layers)  # normalize for training stability


class WaveNet_regressor(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers, dilation_cycle):
        super(WaveNet_regressor, self).__init__()
       
        # initial conv1x1 with relu
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
        
        # all residual layers
        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             dilation_cycle=dilation_cycle)
        
        # final conv1x1 -> relu -> zeroconv1x1
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv1d(skip_channels, out_channels,kernel_size=1))
        self.proj1 = nn.Linear(res_channels,res_channels)

    def forward(self, input_data):
  
 
     
        x = input_data   
        x = x.transpose(-1,-2) 
        x = self.init_conv(x)   
 
        x = self.residual_layer(x)
        x = self.final_conv(x)
        x = x.transpose(-1,-2)
        return x
        


if __name__ == '__main__':

    
    net =  WaveNet_regressor( in_channels=64, res_channels=128, skip_channels=128, out_channels=10, num_res_layers=36, dilation_cycle=12)
    indata = torch.randn((2,640,64), dtype=torch.float32) 
     
     
    
    out = net(indata) 
     
    print(out.shape)
    