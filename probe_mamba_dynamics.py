import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入模型和底层的选择性扫描接口
from models.DepMamba import DepMamba
import models.mamba.selective_scan_interface as selective_scan_interface

# ==========================================
# 步骤 1：构建 Monkey Patch，劫持底层 CUDA 状态
# ==========================================
original_cuda_fwd = selective_scan_interface.selective_scan_cuda.fwd
hooked_states = []

def patched_cuda_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
    # 1. 照常调用原始的底层 CUDA 函数
    out, x, *rest = original_cuda_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

    # 2. 核心探针逻辑：提取中间状态
    # 在底层的 C++ 实现中，x 的形状大致为 [batch, dim, seqlen, dstate_something]
    # 根据 SelectiveScanFn 的源码，真实状态存在于 x[:, :, :, 1::2] 中
    ht_seq = x[:, :, :, 1::2].detach().cpu()
    hooked_states.append(ht_seq)

    # 3. 原样返回，不影响正常前向传播
    return out, x, *rest

# 实施劫持替换
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
        'bidirectional': True # 双向 Mamba
    }
}
model = DepMamba(**config).cuda().eval()

# ==========================================
# 步骤 3：构造时序病理测试用例 (超长冗余序列)
# ==========================================
batch_size = 1
normal_T = 400
stretch_factor = 5
stretched_T = normal_T * stretch_factor # 2000 帧的长序列

# 构造数据：前 400 帧为有效高频信号，后 1600 帧完全是 Padding 或重复噪声
valid_signal = torch.randn(batch_size, normal_T, 136 + 128)
redundant_signal = torch.zeros(batch_size, stretched_T - normal_T, 136 + 128)
pathology_input = torch.cat([valid_signal, redundant_signal], dim=1).cuda()
mask = torch.ones(batch_size, stretched_T).cuda()

print(f"[Probe] 开始向 DepMamba 注入长度为 {stretched_T} 的病理序列...")

# ==========================================
# 步骤 4：执行探针前向传播
# ==========================================
with torch.no_grad():
    _ = model(pathology_input, mask)

print(f"[Probe] 成功截获底层状态！共捕获到 {len(hooked_states)} 次 SSM 算子调用。")

# ==========================================
# 步骤 5：计算动态饱和度 (Delta t)
# ==========================================
# 因为模型使用了 Bi-Mamba，一次层的前向包含音频/视频的前向与反向扫描。
# 我们取第一次截获的状态（例如：CoSSM 中的音频前向扫描状态）进行解剖
target_state_seq = hooked_states[0] # 形状: [Batch, Dim, SeqLen, DState]

# 提取 Batch 0，并排列为 [SeqLen, Dim * DState] 以计算整体范数
ht = target_state_seq[0].permute(1, 0, 2).reshape(stretched_T, -1)

# 计算 t 和 t-1 的状态差异
ht_minus_1 = ht[:-1, :]
ht_current = ht[1:, :]

# 计算范数变化率 (加入 1e-8 防止除 0)
diff_norm = torch.norm(ht_current - ht_minus_1, p=2, dim=-1)
base_norm = torch.norm(ht_minus_1, p=2, dim=-1)
delta_t = diff_norm / (base_norm + 1e-8)
delta_t = delta_t.numpy()

# ==========================================
# 步骤 6：生成诊断图谱
# ==========================================
plt.figure(figsize=(12, 5))
plt.plot(range(1, stretched_T), delta_t, label=r'$\\delta_t = \\frac{||h_t - h_{t-1}||_2}{||h_{t-1}||_2}$', color='b', linewidth=1.5)

# 标记正常输入区和冗余病理区
plt.axvspan(0, normal_T, color='green', alpha=0.1, label='Normal Signal Zone')
plt.axvspan(normal_T, stretched_T, color='red', alpha=0.1, label='Redundant Padding Zone')

# 绘制极小阈值线 (例如 1e-4)
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
