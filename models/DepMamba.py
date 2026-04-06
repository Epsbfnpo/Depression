import warnings
from dataclasses import dataclass
from typing import List, Optional
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (MultiheadAttention, PositionalwiseFeedForward, RelPosMHAXL,)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from mamba_ssm import Mamba
from .mamba.bimamba import Mamba as BiMamba 
from .mamba.mm_bimamba import Mamba as MMBiMamba 
from .base import BaseNet

class BottleneckFusion(nn.Module):
    def __init__(self, d_model=256, num_bottlenecks=4, nhead=4, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.num_bottlenecks = num_bottlenecks
        self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, d_model))
        self.cross_attn_a = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_v = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.alpha_a = nn.Parameter(torch.tensor(1.0))
        self.alpha_v = nn.Parameter(torch.tensor(1.0))
        self.forward_called = False

    def forward(self, xa, xv, padding_mask=None):
        self.forward_called = True
        b_size = xa.size(0)
        key_padding_mask = None
        if padding_mask is not None:
            key_padding_mask = padding_mask == 0
        b_tokens = self.bottlenecks.expand(b_size, -1, -1)
        b_tokens_norm = self.norm_b(b_tokens)
        xa_norm = self.norm_a(xa)
        xv_norm = self.norm_v(xv)

        b_a, _ = self.cross_attn_a(
            query=b_tokens_norm,
            key=xa_norm,
            value=xa_norm,
            key_padding_mask=key_padding_mask,
        )
        b_v, _ = self.cross_attn_v(
            query=b_tokens_norm,
            key=xv_norm,
            value=xv_norm,
            key_padding_mask=key_padding_mask,
        )
        b_fused = b_tokens + (self.alpha_a * b_a) + (self.alpha_v * b_v)
        b_out = b_fused + self.ffn(self.norm_ffn(b_fused))
        return b_out

class CNNEncoderLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0, causal=False, dilation=1,):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.bn1, self.drop)
        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        out = out+x
        return out

class MMCNNEncoderLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0, causal=False, dilation=1,):
        super().__init__()
        self.a_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.a_bn = nn.BatchNorm1d(output_size)
        self.v_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.v_bn = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        self.a_drop = nn.Dropout(dropout)
        self.v_drop = nn.Dropout(dropout)
        self.a_net = nn.Sequential(self.a_conv, self.a_bn, self.a_drop)
        self.v_net = nn.Sequential(self.v_conv, self.v_bn, self.v_drop)
        if input_size != output_size:
            self.a_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
            self.v_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.a_skipconv = None
            self.v_skipconv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_conv.weight.data)
        nn.init.xavier_uniform_(self.v_conv.weight.data)

    def forward(self, xa, xv):
        a_out = self.a_net(xa)
        v_out = self.v_net(xv)
        if self.a_skipconv is not None:
            xa = self.a_skipconv(xa)
        if self.v_skipconv is not None:
            xv = self.v_skipconv(xv)
        a_out = self.relu(a_out + xa)
        v_out = self.relu(v_out + xv)
        return a_out, v_out

class MambaEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, activation='Swish', dropout=0.0, causal=False, mamba_config=None):
        super().__init__()
        assert mamba_config != None
        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish
        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(d_model=d_model, **mamba_config)
        else:
            self.mamba = BiMamba(d_model=d_model, bimamba_type='v2', **mamba_config)
        mamba_config['bidirectional'] = bidirectional
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, inference_params = None):
        mamba_out = self.mamba(x, inference_params)
        out = x + self.drop(self.norm1(mamba_out))
        return out

class MMMambaEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, activation='Swish', dropout=0.0, causal=False, mamba_config=None):
        super().__init__()
        assert mamba_config != None
        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish
        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(d_model=d_model, **mamba_config)
        else:
            self.mamba = MMBiMamba(d_model=d_model, bimamba_type='v2', **mamba_config)
        mamba_config['bidirectional'] = bidirectional
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.a_downsample = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8), nn.BatchNorm1d(d_model),)

    def forward(self, a_x, v_x, a_inference_params = None, v_inference_params = None):
        a_out1, v_out1 = self.mamba(a_x, v_x,a_inference_params,v_inference_params)
        a_out = a_x + self.drop(self.norm1(a_out1))
        v_out = v_x + self.drop(self.norm2(v_out1))
        return a_out, v_out

class CoSSM(nn.Module):
    def __init__(self, num_layers, input_size, output_sizes=[256,512,512], d_ffn=1024, activation='Swish', dropout=0.0, kernel_size = 3, causal=False, mamba_config=None):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size
        cnn_list = []
        a_mamba_list = []
        v_mamba_list = []
        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(input_size = input_size if i<1 else output_sizes[i-1], output_size = output_sizes[i], dropout=dropout))
            a_mamba_list.append(MambaEncoderLayer(d_model=output_sizes[i], d_ffn=d_ffn, dropout=dropout, activation=activation, causal=causal, mamba_config=mamba_config,))
            v_mamba_list.append(MambaEncoderLayer(d_model=output_sizes[i], d_ffn=d_ffn, dropout=dropout, activation=activation, causal=causal, mamba_config=mamba_config,))
        self.a_mamba_layers = torch.nn.ModuleList(a_mamba_list)
        self.v_mamba_layers = torch.nn.ModuleList(v_mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)

    def forward(self, a_x, v_x, a_inference_params = None, v_inference_params = None):
        a_out = a_x
        v_out = v_x
        for cnn_layer, a_mamba_layer, v_mamba_layer in zip(self.cnn_layers, self.a_mamba_layers, self.v_mamba_layers):
            a_out, v_out  = cnn_layer(a_out.permute(0,2,1), v_out.permute(0,2,1))
            a_out = a_out.permute(0,2,1)
            v_out = v_out.permute(0,2,1)
            a_out = a_mamba_layer(a_out, inference_params=a_inference_params)
            v_out = v_mamba_layer(v_out, inference_params=v_inference_params)
        return a_out, v_out

class EnSSM(nn.Module):
    def __init__(self, num_layers, input_size, output_sizes=[256,512,512], d_ffn=1024, activation='Swish', dropout=0.0, causal=False, mamba_config=None):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size
        cnn_list = []
        mamba_list = []
        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(input_size = input_size if i<1 else output_sizes[i-1], output_size = output_sizes[i], dropout=dropout))
            mamba_list.append(MambaEncoderLayer(d_model=output_sizes[i], d_ffn=d_ffn, dropout=dropout, activation=activation, causal=causal, mamba_config=mamba_config,))
        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)

    def forward(self, x, inference_params = None,):
        out = x
        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out  = cnn_layer(out.permute(0,2,1))
            out = out.permute(0,2,1)
            out = mamba_layer(out, inference_params = inference_params,)
        return out

class DepMamba(BaseNet):
    def __init__(self, audio_input_size=161, video_input_size=161, mm_input_size=128, mm_output_sizes=[256,64], d_ffn=1024, num_layers=8, dropout=0.1, activation='Swish', causal=False, mamba_config=None):
        super(DepMamba, self).__init__()
        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.cossm_encoder = CoSSM(
            num_layers=num_layers,
            input_size=mm_input_size,
            output_sizes=[mm_input_size] * num_layers,
            d_ffn=d_ffn,
            activation=activation,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config
        )
        self.bottleneck_fusion = BottleneckFusion(
            d_model=mm_input_size,
            num_bottlenecks=2,
            nhead=4,
            dropout=max(dropout, 0.3)
        )
        self.output = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(mm_input_size, mm_input_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mm_input_size // 2, 2)
        )
        self.feature_extractor_called = False
        self.time_mask_prob = 0.3
        self.time_mask_width = 50
        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)

    def _apply_time_masking(self, x, padding_mask=None):
        if (not self.training) or (self.time_mask_prob <= 0):
            return x
        b_size, seq_len, _ = x.shape
        mask_len = min(self.time_mask_width, seq_len)
        if mask_len <= 0:
            return x
        aug_x = x.clone()
        for b in range(b_size):
            if torch.rand(1, device=x.device).item() >= self.time_mask_prob:
                continue
            valid_len = seq_len
            if padding_mask is not None:
                valid_len = int(padding_mask[b].sum().item())
            if valid_len <= 1:
                continue
            current_len = min(mask_len, valid_len)
            max_start = max(valid_len - current_len, 0)
            start = 0 if max_start == 0 else torch.randint(0, max_start + 1, (1,), device=x.device).item()
            aug_x[b, start:start + current_len, :] = 0.0
        return aug_x

    def feature_extractor(self, x, padding_mask=None, a_inference_params = None, v_inference_params = None):
        self.feature_extractor_called = True
        xa = x[:, :, 136:]
        xv = x[:, :, :136]
        xa = self.conv_audio(xa.permute(0,2,1)).permute(0,2,1)
        xv = self.conv_video(xv.permute(0,2,1)).permute(0,2,1)
        xa_out, xv_out = self.cossm_encoder(xa, xv, a_inference_params, v_inference_params)
        b_out = self.bottleneck_fusion(xa_out, xv_out, padding_mask)
        global_feature = b_out.mean(dim=1)
        return global_feature

    def classifier(self, x):
        return self.output(x)

    def reset_silent_failure_flags(self):
        self.feature_extractor_called = False
        self.bottleneck_fusion.forward_called = False

    def assert_new_path_executed(self):
        if not self.feature_extractor_called:
            raise RuntimeError("Silent failure detected: DepMamba.feature_extractor was not executed.")
        if not self.bottleneck_fusion.forward_called:
            raise RuntimeError("Silent failure detected: BottleneckFusion.forward was not executed.")

    def forward(self, x, padding_mask=None):
        x = self._apply_time_masking(x, padding_mask)
        x = self.feature_extractor(x, padding_mask)
        out = self.output(x)
        return out
