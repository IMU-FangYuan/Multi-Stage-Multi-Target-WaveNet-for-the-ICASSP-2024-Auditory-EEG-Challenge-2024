import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
input_length = 3650


# Depth-Wise Conv1D block from Conv-TasNet
class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True):
        super(DepthConv1d, self).__init__()

        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()

        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))


        output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual

        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class ConvCrossAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, in_ch, kernel_size, dilation, dropout=0.1):
        super(ConvCrossAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v


        self.w_qs = DepthConv1d(kernel=kernel_size, input_channel=in_ch, hidden_channel=in_ch*2,
                              dilation=dilation, padding='same')
        self.w_ks = DepthConv1d(kernel=kernel_size, input_channel=in_ch, hidden_channel=in_ch*2,
                              dilation=dilation, padding='same')
        self.w_vs = DepthConv1d(kernel=kernel_size, input_channel=in_ch, hidden_channel=in_ch*2,
                              dilation=dilation, padding='same')
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.GroupNorm(1, in_ch, eps=1e-08)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        #q k v  [b t f]
        residual = v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q, _ = self.w_qs(q)
        k, _ = self.w_ks(k)
        v, _ = self.w_vs(v)
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(sz_b, len_v, -1)
        out = self.dropout(out)
        out += residual

        out = self.layer_norm(out)

        return out

class MultiLayerCrossAttention(nn.Module):
    def __init__(self, input_size, layer, in_ch, kernel_size, dilation):
        super(MultiLayerCrossAttention, self).__init__()
        self.audio_encoder = nn.ModuleList()
        self.spike_encoder = nn.ModuleList()
        self.layer = layer
        self.projection = nn.Conv1d(in_ch*4, in_ch, kernel_size, padding='same')
        self.LayernormList_audio = nn.ModuleList()
        self.LayernormList_spike = nn.ModuleList()
        self.layernorm_out = nn.GroupNorm(1, in_ch, eps=1e-08)
        for i in range(layer):
            self.LayernormList_audio.append(nn.GroupNorm(1, in_ch, eps=1e-08))
            self.LayernormList_spike.append(nn.GroupNorm(1, in_ch, eps=1e-08))
        for i in range(layer):
            self.audio_encoder.append(ConvCrossAttention(n_head=1, d_model=input_size, d_k=input_size, d_v=input_size,
                                                  in_ch=in_ch, kernel_size=kernel_size, dilation=dilation))
            self.spike_encoder.append(ConvCrossAttention(n_head=1, d_model=input_size, d_k=input_size, d_v=input_size,
                                                         in_ch=in_ch, kernel_size=kernel_size, dilation=dilation))


    def forward(self, audio, spike):
        out_audio = audio
        out_spike = spike
        skip_audio = 0.
        skip_spike = 0.
        residual_audio = audio
        residual_spike = spike
       
        for i in range(self.layer):
            out_audio = self.audio_encoder[i](out_spike, out_audio, out_audio)
            out_spike = self.spike_encoder[i](out_audio, out_spike, out_spike)
            out_audio = out_audio + residual_audio
            out_audio = self.LayernormList_audio[i](out_audio)
            out_spike = out_spike + residual_spike
            out_spike = self.LayernormList_spike[i](out_spike)
            residual_audio = out_audio
            residual_spike = out_spike
            skip_audio += out_audio
            skip_spike += out_spike
        out = torch.cat((skip_audio, audio, out_spike, spike), dim=1)
        out = self.projection(out)
        out = self.layernorm_out(out)
        return out        


# The separation network adapted from Conv-TasNet
class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, CMCA_layer_num=3, skip=True,
                 dilated=True):
        super(TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        # normalization

        self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)


        self.BN = nn.Conv1d(input_dim, BN_dim, 1)
        self.fusion = MultiLayerCrossAttention(input_size=input_length, layer=CMCA_layer_num, in_ch=BN_dim, kernel_size=kernel,
                                               dilation=1)
        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated
        self.layer = layer
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip))
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
                    
       
        
        # output layer
        
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                   )
        
        self.skip = skip
        
    def forward(self, input, spike):
        
        # input shape: (B, N, L)
        
        # normalization
        output = self.BN(self.LN(input))
        
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
                if i == self.layer:
                    spike = F.pad(spike, (0, (output.size(2) - spike.size(2))))
                    output = self.fusion(output, spike)

        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual
            
        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)
        
        return output