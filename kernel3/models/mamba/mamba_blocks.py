import math
import torch
import torch.nn as nn
from functools import partial
from mamba_ssm import Mamba
from modules.mamba.bimamba import Mamba as BiMamba 
from modules.mamba.bimamba import Block as PreNormBlock

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def create_block(d_model, ssm_cls=None, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False, fused_add_norm=True, layer_idx=None, device=None, dtype=None,):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(ssm_cls, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = PreNormBlock(d_model, mixer_cls, norm_cls=norm_cls, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32,)
    block.layer_idx = layer_idx
    return block

def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1,):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class LnMambaAdd(nn.Module):
    def __init__(self, d_model, ssm_cls, ssm_cfg, rms_norm=False, layer_idx=None):
        super().__init__()
        if rms_norm:
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)
        self.mamba = ssm_cls(d_model=d_model, **ssm_cfg)
        print(type(self.mamba))
        print('Created LnMambaAdd.')

    def forward(self, x, residual=None, inference_params=None):
        if residual != None:
            x = x + residual
        return self.mamba(self.norm(x)), x

class MambaBlocksSequential(nn.Module):
    def __init__(self, n_mamba: int, bidirectional: bool, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4, dt_rank: str="auto", conv_bias: bool = True, bias: bool = False, fused_add_norm: bool = True, rms_norm: bool = False, norm_epsilon: float = 1e-5, initializer_cfg=None, residual_in_fp32=False, use_simple_block=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.bidirectional = bidirectional
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        self.use_simple_block = use_simple_block
        ssm_cfg = {"d_state": d_state, "expand": expand, "d_conv": d_conv, "dt_rank": dt_rank, "conv_bias": conv_bias, "bias": bias}
        if bidirectional:
            ssm_cfg["bimamba_type"] = "v2"
        if use_simple_block:
            self.layers = nn.Sequential(*[LnMambaAdd(d_model=d_model, ssm_cls=BiMamba if bidirectional else Mamba, ssm_cfg=ssm_cfg, rms_norm=rms_norm, layer_idx=i) for i in range(n_mamba)])
        else:
            self.layers = nn.Sequential(*[create_block(d_model=d_model, ssm_cls=BiMamba if bidirectional else Mamba, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, layer_idx=i,) for i in range(n_mamba)])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)
        self.apply(partial(_init_weights, n_layer=n_mamba, **(initializer_cfg if initializer_cfg is not None else {}),))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) for i, layer in enumerate(self.layers)}
    
    def forward(self, x, inference_params=None):
        hidden_states = x
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(hidden_states, self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps, residual=residual, prenorm=False, residual_in_fp32=self.residual_in_fp32,)
        return hidden_states
        