import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 导入模型
from models.DepMamba import DepMamba

def run_ultimate_erf_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device} 进行多点时序探针诊断...")

    # 1. 加载模型配置
    config = {
        'audio_input_size': 25,
        'video_input_size': 136,
        'mm_input_size': 256,
        'mm_output_sizes': [256],
        'dropout': 0.1,
        'd_ffn': 1024,
        'num_layers': 1,
        'activation': 'GELU',
        'causal': False,
        'mamba_config': {
            'd_state': 12,
            'expand': 4,
            'd_conv': 4,
            'bidirectional': True # 开启双向
        }
    }

    model = DepMamba(**config).to(device)
    model.eval()

    # 2. 准备基准序列
    seq_len = 1000
    X = torch.randn(1, seq_len, 161, device=device) * 0.1
    X.requires_grad_(True)
    mask = torch.ones(1, seq_len, device=device)

    # 3. 注册 Hook 拦截池化前特征
    hook_data = {}
    def hook_fn(module, input, output):
        hook_data['pre_pool_features'] = output

    hook_handle = model.enssm_encoder.register_forward_hook(hook_fn)

    # 4. 执行前向传播
    y_logits = model(X, mask)
    pre_pool_feat = hook_data['pre_pool_features'] # shape: [1, 1000, Dim]

    # --- 辅助函数：计算并提取特定 Target 的梯度范数 ---
    def get_erf_for_target(target_tensor):
        model.zero_grad()
        if X.grad is not None:
            X.grad.zero_()

        # retain_graph=True 允许我们对同一个计算图多次 backward
        target_tensor.backward(retain_graph=True)

        grad_wrt_X = X.grad.clone()
        # 计算每个时间步的 Frobenius 范数
        erf = torch.norm(grad_wrt_X.squeeze(0), p='fro', dim=1).cpu().numpy()
        return erf

    # 5. 执行多点探针诊断
    print("正在计算全局池化系统的 ERF...")
    I_sys = get_erf_for_target(y_logits.sum())

    print("正在计算 t=0 (验证反向 Mamba 回溯) 的 ERF...")
    I_t0 = get_erf_for_target(pre_pool_feat[:, 0, :].sum())

    print("正在计算 t=500 (验证双向扩散) 的 ERF...")
    I_t500 = get_erf_for_target(pre_pool_feat[:, 500, :].sum())

    print("正在计算 t=999 (验证正向 Mamba 记忆) 的 ERF...")
    I_t999 = get_erf_for_target(pre_pool_feat[:, -1, :].sum())

    hook_handle.remove()

    # 6. 可视化终极确诊报告 (2x2 画布)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Ultimate ERF Probe: Unveiling Bidirectional Mamba Dynamics", fontsize=18, fontweight='bold')

    # 图 1: 系统级 (带池化掩护)
    axs[0, 0].plot(I_sys, color='blue', linewidth=2)
    axs[0, 0].fill_between(range(seq_len), I_sys, alpha=0.2, color='blue')
    axs[0, 0].set_title("1. System-Level ERF (With Global Pooling)", fontsize=12)
    axs[0, 0].set_ylabel(r"$|| \partial y / \partial x_t ||_F$", fontsize=11)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 图 2: 起点探测 t=0
    axs[0, 1].plot(I_t0, color='purple', linewidth=2)
    axs[0, 1].fill_between(range(seq_len), I_t0, alpha=0.2, color='purple')
    axs[0, 1].axvline(x=0, color='black', linestyle=':', label='Probe Target t=0')
    axs[0, 1].set_title("2. Probe at $t=0$ (Testing Backward Mamba)", fontsize=12)
    axs[0, 1].set_ylabel(r"$|| \partial h_0 / \partial x_t ||_F$", fontsize=11)
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 图 3: 中点探测 t=500
    axs[1, 0].plot(I_t500, color='green', linewidth=2)
    axs[1, 0].fill_between(range(seq_len), I_t500, alpha=0.2, color='green')
    axs[1, 0].axvline(x=500, color='black', linestyle=':', label='Probe Target t=500')
    axs[1, 0].set_title("3. Probe at $t=500$ (Testing Bidirectional Spread)", fontsize=12)
    axs[1, 0].set_xlabel("Input Time Step $t$", fontsize=11)
    axs[1, 0].set_ylabel(r"$|| \partial h_{500} / \partial x_t ||_F$", fontsize=11)
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 图 4: 终点探测 t=999
    axs[1, 1].plot(I_t999, color='red', linewidth=2)
    axs[1, 1].fill_between(range(seq_len), I_t999, alpha=0.2, color='red')
    axs[1, 1].axvline(x=999, color='black', linestyle=':', label='Probe Target t=999')
    axs[1, 1].set_title("4. Probe at $t=999$ (Testing Forward Mamba)", fontsize=12)
    axs[1, 1].set_xlabel("Input Time Step $t$", fontsize=11)
    axs[1, 1].set_ylabel(r"$|| \partial h_{999} / \partial x_t ||_F$", fontsize=11)
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("ultimate_mamba_erf.png", dpi=300)
    print("确诊报告已生成：ultimate_mamba_erf.png")

if __name__ == "__main__":
    run_ultimate_erf_probe()
