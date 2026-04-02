import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入模型和底层的选择性扫描接口
from models.DepMamba import DepMamba
import models.mamba.selective_scan_interface as selective_scan_interface

# ==========================================
# 核心重演引擎：在 CPU 上重构 Mamba 时序动力学
# ==========================================
def extract_hidden_states_pytorch(u, delta, A, B, C, delta_bias, delta_softplus):
    """
    通过拦截的输入，用纯 PyTorch 重演 Selective Scan，捕获每一帧的 h_t
    """
    # 转移到 CPU 并转为 float32 防止精度溢出
    u = u.detach().cpu().float()
    delta = delta.detach().cpu().float()
    A = A.detach().cpu().float()
    B = B.detach().cpu().float()
    if delta_bias is not None:
        delta_bias = delta_bias.detach().cpu().float()

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    # 1. 计算离散化步长 \Delta
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    h = torch.zeros(batch, dim, dstate)
    h_seq = []

    # 2. 逐步推演动力学方程: h_t = dA * h_{t-1} + dB * x_t
    for t in range(seqlen):
        dt = delta[:, :, t] # [B, D]
        dA = torch.exp(dt.unsqueeze(-1) * A) # [B, D, N]
        
        # B 的形状可能是 [B, N, L] 或 [B, G, N, L]
        if B.dim() == 3:
            B_t = B[:, :, t]
            dB = dt.unsqueeze(-1) * B_t.unsqueeze(1) # [B, D, N]
        else:
            B_t = B[:, 0, :, t]
            dB = dt.unsqueeze(-1) * B_t.unsqueeze(1)

        u_t = u[:, :, t].unsqueeze(-1) # [B, D, 1]
        
        h = dA * h + dB * u_t # 更新隐藏状态
        h_seq.append(h.clone())

    # 返回形状: [Batch, Dim, SeqLen, DState]
    return torch.stack(h_seq, dim=2)


# ==========================================
# 步骤 1：构建 Monkey Patch，劫持并重演
# ==========================================
original_cuda_fwd = selective_scan_interface.selective_scan_cuda.fwd
hooked_states = []

def patched_cuda_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
    # 1. 拦截输入，在 CPU 上重演动力学，提取全量时序状态 ht
    ht_seq = extract_hidden_states_pytorch(u, delta, A, B, C, delta_bias, delta_softplus)
    hooked_states.append(ht_seq)
    
    # 2. 原样调用 CUDA 算子，保证模型主流程不受任何影响
    return original_cuda_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

selective_scan_interface.selective_scan_cuda.fwd = patched_cuda_fwd

# ==========================================
# 步骤 2：初始化模型 (以 LMVD 配置为例)
# ==========================================
config = {
    'audio_input_size': 128,
    'video_input_size': 136,
    'mm_input_size': 256,
    'mm_output_sizes': [256],
    'dropout': 0.1,
    'd_ffn': 1024,
    'num_layers': 1,
    'activation': 'GELU',
    'causal': False,
    'mamba_config': {
        'd_state': 16,
        'expand': 4,
        'd_conv': 4,
        'bidirectional': True
    }
}
model = DepMamba(**config).cuda().eval()

# ==========================================
# 步骤 3：构造时序病理测试用例 (三明治结构探针)
# ==========================================
batch_size = 1
normal_T = 400
stretch_factor = 5
stretched_T = normal_T * stretch_factor # 2000 帧长序列

# 构造三明治数据：首尾是有效信号，中间是巨大的冗余黑洞
valid_signal_start = torch.randn(batch_size, normal_T, 136 + 128)
redundant_signal = torch.zeros(batch_size, stretched_T - normal_T - 200, 136 + 128) # 1400 帧冗余
valid_signal_end = torch.randn(batch_size, 200, 136 + 128) # 末尾 200 帧的“强心针”唤醒刺激！

pathology_input = torch.cat([valid_signal_start, redundant_signal, valid_signal_end], dim=1).cuda()
mask = torch.ones(batch_size, stretched_T).cuda()

print(f"[Probe] 开始向 DepMamba 注入含有唤醒刺激的三明治病理序列...")

with torch.no_grad():
    _ = model(pathology_input, mask)

print(f"[Probe] 成功截获底层状态！共捕获到 {len(hooked_states)} 次 SSM 算子调用。")

# ==========================================
# 步骤 4：计算动态饱和度 (Delta t)
# ==========================================
# target_state_seq 形状现在一定是: [1, Dim, 2000, DState]
target_state_seq = hooked_states[0] 

# 取 Batch 0 -> [Dim, SeqLen, DState]
# permute(1, 0, 2) -> [SeqLen, Dim, DState]
# reshape(2000, -1) -> [2000, Dim * DState]
ht = target_state_seq[0].permute(1, 0, 2).reshape(stretched_T, -1)

# 计算 t 和 t-1 的差异
ht_minus_1 = ht[:-1, :]
ht_current = ht[1:, :]

# 计算范数变化率
diff_norm = torch.norm(ht_current - ht_minus_1, p=2, dim=-1)
base_norm = torch.norm(ht_minus_1, p=2, dim=-1)
delta_t = diff_norm / (base_norm + 1e-8)
delta_t = delta_t.numpy()

# ==========================================
# 步骤 5：生成诊断图谱
# ==========================================
plt.figure(figsize=(12, 5))
plt.plot(range(1, stretched_T), delta_t, label='Delta_t (Norm Change Rate)', color='b', linewidth=1.5)

plt.axvspan(0, normal_T, color='green', alpha=0.1, label='Normal Signal Zone')
plt.axvspan(normal_T, stretched_T, color='red', alpha=0.1, label='Redundant Padding Zone')
plt.axvspan(1800, 2000, color='orange', alpha=0.2, label='Wake-up Signal Zone')

collapse_threshold = 1e-4
plt.axhline(y=collapse_threshold, color='r', linestyle='--', label=f'Collapse Threshold ({collapse_threshold})')

plt.title("DepMamba Internal Dynamics: State Saturation Probe")
plt.xlabel("Time Step (t)")
plt.ylabel("State Change Rate (Log Scale)")
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("mamba_state_probe.png", dpi=300)
print("[Probe] 诊断图谱已保存至 mamba_state_probe.png")
