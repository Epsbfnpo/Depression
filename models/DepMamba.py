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

class CNNEncoderLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0, causal=False, dilation=1,):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)
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
        self.a_net = nn.Sequential(self.a_conv, self.a_bn, self.relu, self.a_drop)
        self.v_net = nn.Sequential(self.v_conv, self.v_bn, self.relu, self.v_drop)
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
        a_out = a_out+xa
        v_out = v_out+xv
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
        out = x + self.norm1(self.mamba(x, inference_params))
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
        a_out = a_x + self.norm1(a_out1)
        v_out = v_x + self.norm2(v_out1)
        return a_out, v_out

class CoSSM(nn.Module):
    def __init__(self, num_layers, input_size, output_sizes=[256,512,512], d_ffn=1024, activation='Swish', dropout=0.0, kernel_size = 3, causal=False, mamba_config=None):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size
        cnn_list = []
        mamba_list = []
        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(input_size = input_size if i<1 else output_sizes[i-1], output_size = output_sizes[i], dropout=dropout))
            mamba_list.append(MMMambaEncoderLayer(d_model=output_sizes[i], d_ffn=d_ffn, dropout=dropout, activation=activation, causal=causal, mamba_config=mamba_config,))
        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)

    def forward(self, a_x, v_x, a_inference_params = None, v_inference_params = None):
        a_out = a_x
        v_out = v_x
        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            a_out, v_out  = cnn_layer(a_out.permute(0,2,1), v_out.permute(0,2,1))
            a_out = a_out.permute(0,2,1)
            v_out = v_out.permute(0,2,1)
            a_out, v_out = mamba_layer(a_out, v_out, a_inference_params = a_inference_params, v_inference_params = v_inference_params)
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

class MoERouter(nn.Module):
    """Dynamic router that predicts per-sample expert weights."""

    def __init__(self, input_dim, num_experts=3):
        super().__init__()
        hidden_dim = max(1, input_dim // 2)
        self.router_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x, padding_mask=None):
        # x: [B, L, C]
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()
            x_pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            x_pooled = x.mean(dim=1)
        logits = self.router_network(x_pooled)
        logits = logits / (torch.norm(logits, dim=-1, keepdim=True) + 1e-8)
        temperature = 1.5
        return F.softmax(logits / temperature, dim=-1)

class MoEEnSSM(nn.Module):
    """MoE-enhanced fusion module with unimodal auxiliary heads."""

    def __init__(self, d_model_single, d_model_concat, num_layers, d_ffn, activation='Swish', dropout=0.0, causal=False, mamba_config=None):
        super().__init__()
        self.router = MoERouter(input_dim=d_model_concat, num_experts=3)
        self.expert_audio = EnSSM(num_layers, d_model_single, [d_model_single], d_ffn, activation=activation, dropout=dropout, causal=causal, mamba_config=mamba_config)
        self.expert_video = EnSSM(num_layers, d_model_single, [d_model_single], d_ffn, activation=activation, dropout=dropout, causal=causal, mamba_config=mamba_config)
        self.expert_fusion = EnSSM(num_layers, d_model_concat, [d_model_concat], d_ffn, activation=activation, dropout=dropout, causal=causal, mamba_config=mamba_config)
        self.proj_a = nn.Linear(d_model_single, d_model_concat)
        self.proj_v = nn.Linear(d_model_single, d_model_concat)
        self.layer_norm = LayerNorm(d_model_concat, eps=1e-6)
        self.aux_classifier_a = nn.Linear(d_model_single, 1)
        self.aux_classifier_v = nn.Linear(d_model_single, 1)

    def forward(self, xa, xv, padding_mask=None):
        x_concat = torch.cat([xa, xv], dim=-1)
        weights = self.router(x_concat, padding_mask)
        w_a = weights[:, 0].unsqueeze(1).unsqueeze(2)
        w_v = weights[:, 1].unsqueeze(1).unsqueeze(2)
        w_f = weights[:, 2].unsqueeze(1).unsqueeze(2)
        self.dbg_w_a = weights[:, 0].mean().item()
        self.dbg_w_v = weights[:, 1].mean().item()
        self.dbg_w_f = weights[:, 2].mean().item()

        y_a = self.expert_audio(F.dropout(xa, p=0.3, training=self.training))
        y_v = self.expert_video(xv)
        y_f = self.expert_fusion(x_concat)

        y_a_proj = self.proj_a(y_a)
        y_v_proj = self.proj_v(y_v)
        out_fused = self.layer_norm(w_a * y_a_proj + w_v * y_v_proj + w_f * y_f)

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()
            pool_a = (y_a * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            pool_v = (y_v * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            pool_a = y_a.mean(dim=1)
            pool_v = y_v.mean(dim=1)

        aux_logits_a = self.aux_classifier_a(pool_a).squeeze(-1)
        aux_logits_v = self.aux_classifier_v(pool_v).squeeze(-1)
        return out_fused, aux_logits_a, aux_logits_v, weights

class DepMamba(BaseNet):
    def __init__(self, audio_input_size=161, video_input_size=161, mm_input_size=128, mm_output_sizes=[256,64], d_ffn=1024, num_layers=8, dropout=0.1, activation='Swish', causal=False, mamba_config=None):
        super().__init__()
        self.cossm_encoder = CoSSM(num_layers, mm_input_size, mm_output_sizes, d_ffn, activation=activation, dropout=dropout, causal=causal, mamba_config=mamba_config)
        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.moe_enssm = MoEEnSSM(num_layers=num_layers, d_model_single=mm_output_sizes[-1], d_model_concat=mm_output_sizes[-1]*2, d_ffn=d_ffn, activation=activation, dropout=dropout, causal=causal, mamba_config=mamba_config)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier_drop = nn.Dropout(0.5)
        self.output = nn.Linear(mm_output_sizes[-1]*2, 1)
        self.current_aux_a = None
        self.current_aux_v = None
        self.current_weights = None
        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)

    def feature_extractor(self, x, padding_mask=None, a_inference_params = None, v_inference_params = None):
        xa = x[:, :, 136:]
        xv = x[:, :, :136]
        xa = self.conv_audio(xa.permute(0,2,1)).permute(0,2,1)
        xv = self.conv_video(xv.permute(0,2,1)).permute(0,2,1)
        xa, xv = self.cossm_encoder(xa, xv, a_inference_params, v_inference_params)
        x, aux_a, aux_v, weights = self.moe_enssm(xa, xv, padding_mask)
        self.current_aux_a = aux_a
        self.current_aux_v = aux_v
        self.current_weights = weights
        if padding_mask is not None:
            x = x * (padding_mask.unsqueeze(-1).float())
            x = x.sum(dim=1) / (padding_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=False)
        else:
            x = self.pool(x.permute(0,2,1)).squeeze(-1)
        x = self.classifier_drop(x)
        return x

    def classifier(self, x):
        return self.output(x)

    def forward(self, x, padding_mask=None, **kwargs):
        x = self.feature_extractor(x, padding_mask)
        out = self.output(x).squeeze(-1)
        if self.training:
            return out, self.current_aux_a, self.current_aux_v, self.current_weights
        try:
            w_a = self.moe_enssm.dbg_w_a
            w_v = self.moe_enssm.dbg_w_v
            w_f = self.moe_enssm.dbg_w_f
        except Exception:
            w_a, w_v, w_f = -1.0, -1.0, -1.0
        return out, out.new_tensor(w_a), out.new_tensor(w_v), out.new_tensor(w_f)
