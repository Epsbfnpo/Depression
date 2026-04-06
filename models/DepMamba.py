import math

import torch
import torch.nn as nn

from .base import BaseNet
from .mamba.mamba_blocks import MambaBlock


class TrueBottleneckBlock(nn.Module):
    def __init__(self, d_model, max_len=50000, num_bottlenecks=4, nhead=8):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.latent_pe = nn.Parameter(torch.randn(1, num_bottlenecks, d_model))

        self.latent_query_a = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.latent_query_v = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.gate_a = nn.Linear(d_model, d_model)
        self.gate_v = nn.Linear(d_model, d_model)

        self.latent_self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.latent_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.a_query_latent = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.v_query_latent = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.norm_l1 = nn.LayerNorm(d_model)
        self.norm_l2 = nn.LayerNorm(d_model)
        self.norm_l3 = nn.LayerNorm(d_model)
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_v1 = nn.LayerNorm(d_model)

    def forward(self, xa, xv, latents):
        pe_a = self.pe[:, : xa.size(1), :]
        pe_v = self.pe[:, : xv.size(1), :]

        xa_norm = self.norm_a1(xa)
        xv_norm = self.norm_v1(xv)
        latents_norm = self.norm_l1(latents)

        xa_for_routing = xa_norm + pe_a
        xv_for_routing = xv_norm + pe_v
        latents_for_routing = latents_norm + self.latent_pe

        update_l_from_a, _ = self.latent_query_a(query=latents_for_routing, key=xa_for_routing, value=xa_norm)
        update_l_from_v, _ = self.latent_query_v(query=latents_for_routing, key=xv_for_routing, value=xv_norm)

        gated_a = torch.sigmoid(self.gate_a(update_l_from_a)) * update_l_from_a
        gated_v = torch.sigmoid(self.gate_v(update_l_from_v)) * update_l_from_v
        latents = latents + gated_a + gated_v

        latents_norm2 = self.norm_l2(latents)
        latents_for_self_routing = latents_norm2 + self.latent_pe
        self_attn_l, _ = self.latent_self_attn(
            query=latents_for_self_routing,
            key=latents_for_self_routing,
            value=latents_norm2,
        )
        latents = latents + self_attn_l
        latents = latents + self.latent_ffn(self.norm_l3(latents))

        latents_for_feedback = self.norm_l3(latents)
        latents_for_feedback_routing = latents_for_feedback + self.latent_pe

        a_feedback, _ = self.a_query_latent(
            query=xa_for_routing,
            key=latents_for_feedback_routing,
            value=latents_for_feedback,
        )
        xa = xa + a_feedback

        v_feedback, _ = self.v_query_latent(
            query=xv_for_routing,
            key=latents_for_feedback_routing,
            value=latents_for_feedback,
        )
        xv = xv + v_feedback

        return xa, xv, latents


class DepMamba(BaseNet):
    def __init__(
        self,
        audio_input_size=161,
        video_input_size=161,
        mm_input_size=128,
        num_layers=3,
        num_bottlenecks=4,
        num_classes=2,
        mamba_config=None,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.feature_extractor_called = False
        self.bottleneck_forward_called = False

        self.proj_a = nn.Linear(audio_input_size, mm_input_size)
        self.proj_v = nn.Linear(video_input_size, mm_input_size)

        self.latent_tokens = nn.Parameter(torch.randn(1, num_bottlenecks, mm_input_size))

        cfg = mamba_config or {}
        self.mamba_layers_a = nn.ModuleList(
            [
                MambaBlock(
                    mm_input_size,
                    d_state=cfg.get("d_state", 16),
                    expand=cfg.get("expand", 2),
                    d_conv=cfg.get("d_conv", 4),
                    bidirectional=cfg.get("bidirectional", True),
                )
                for _ in range(num_layers)
            ]
        )
        self.mamba_layers_v = nn.ModuleList(
            [
                MambaBlock(
                    mm_input_size,
                    d_state=cfg.get("d_state", 16),
                    expand=cfg.get("expand", 2),
                    d_conv=cfg.get("d_conv", 4),
                    bidirectional=cfg.get("bidirectional", True),
                )
                for _ in range(num_layers)
            ]
        )

        self.bottleneck_blocks = nn.ModuleList(
            [TrueBottleneckBlock(mm_input_size, num_bottlenecks=num_bottlenecks) for _ in range(num_layers)]
        )

        self.output = nn.Sequential(
            nn.Linear(mm_input_size, mm_input_size // 2),
            nn.GELU(),
            nn.Linear(mm_input_size // 2, num_classes),
        )

    def feature_extractor(self, audio, video):
        self.feature_extractor_called = True
        xa = self.proj_a(audio)
        xv = self.proj_v(video)
        latents = self.latent_tokens.expand(xa.size(0), -1, -1)

        for i in range(self.num_layers):
            xa = self.mamba_layers_a[i](xa)
            xv = self.mamba_layers_v[i](xv)
            xa, xv, latents = self.bottleneck_blocks[i](xa, xv, latents)

        self.bottleneck_forward_called = True
        global_feature = latents.mean(dim=1)
        return global_feature

    def classifier(self, x):
        return self.output(x)

    def reset_silent_failure_flags(self):
        self.feature_extractor_called = False
        self.bottleneck_forward_called = False

    def assert_new_path_executed(self):
        if not self.feature_extractor_called:
            raise RuntimeError("Silent failure detected: DepMamba.feature_extractor was not executed.")
        if not self.bottleneck_forward_called:
            raise RuntimeError("Silent failure detected: TrueBottleneckBlock path was not executed.")

    def forward(self, audio, video):
        x = self.feature_extractor(audio, video)
        out = self.output(x)
        return out
