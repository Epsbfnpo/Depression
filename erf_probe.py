import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import yaml

# 导入模型
from models.DepMamba import DepMamba

def run_erf_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 1. 加载模型配置 (以 config.yaml 中的 dvlog 参数为例)
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
            'bidirectional': True
        }
    }

    # 实例化模型
    model = DepMamba(**config).to(device)
    model.eval() # 探针分析需在 eval 模式下进行，关闭 Dropout 等随机性

    # 2. 准备基准序列 (维度: Batch=1, Seq_len=1000, Feat=161)
    seq_len = 1000
    # 使用微小的高斯噪声而非全零，激活 Mamba 的数据依赖机制
    X = torch.randn(1, seq_len, 161, device=device) * 0.1
    X.requires_grad_(True)
    mask = torch.ones(1, seq_len, device=device)

    # 3. 注册 Forward Hook 拦截池化前的时序特征
    hook_data = {}
    def hook_fn(module, input, output):
        # 拦截 enssm_encoder 的输出，此时尚未经过时序池化
        # output shape: [Batch, Seq_len, Dim]
        hook_data['pre_pool_features'] = output

    hook_handle = model.enssm_encoder.register_forward_hook(hook_fn)

    # 4. 执行前向传播
    y_logits = model(X, mask)

    # ==========================================
    # 诊断维度 A：带池化掩护的系统级响应 (你的原始设想)
    # ==========================================
    y_logits.sum().backward(retain_graph=True)
    # 提取梯度并沿特征维度计算 Frobenius 范数
    grad_y_wrt_X = X.grad.clone()
    I_sys = torch.norm(grad_y_wrt_X.squeeze(0), p='fro', dim=1).cpu().numpy()

    # 清空梯度，为下一个更严谨的探针做准备
    model.zero_grad()
    X.grad.zero_()

    # ==========================================
    # 诊断维度 B：纯 Mamba 动力学响应 (剥离池化层)
    # ==========================================
    pre_pool_feat = hook_data['pre_pool_features']
    # 目标：抽取最后一个时间步 (t=L) 的特征向量的范数
    last_step_target = pre_pool_feat[:, -1, :].sum()

    # 反向传播求导：计算最后一个状态对所有历史输入 X_t 的依赖
    last_step_target.backward()
    grad_last_wrt_X = X.grad.clone()
    I_mamba = torch.norm(grad_last_wrt_X.squeeze(0), p='fro', dim=1).cpu().numpy()

    hook_handle.remove()

    # 5. 可视化确诊报告
    plt.figure(figsize=(14, 6))

    # 绘制图 A
    plt.subplot(1, 2, 1)
    plt.plot(I_sys, color='blue', linewidth=2)
    plt.title("Diagnosis A: System-Level ERF (With Pooling)", fontsize=12)
    plt.xlabel("Input Time Step $t$", fontsize=11)
    plt.ylabel(r"Influence $I(t) = || \partial y / \partial x_t ||_F$", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.fill_between(range(seq_len), I_sys, alpha=0.2, color='blue')

    # 绘制图 B
    plt.subplot(1, 2, 2)
    plt.plot(I_mamba, color='red', linewidth=2)
    plt.title("Diagnosis B: Mamba Core Dynamics (Last Step Target)", fontsize=12)
    plt.xlabel("Input Time Step $t$", fontsize=11)
    plt.ylabel(r"Influence $I(t) = || \partial h_L / \partial x_t ||_F$", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.fill_between(range(seq_len), I_mamba, alpha=0.2, color='red')

    plt.suptitle("Mamba Effective Receptive Field (ERF) Probe", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("mamba_erf_diagnosis.png", dpi=300)
    print("确诊报告已生成：mamba_erf_diagnosis.png")

if __name__ == "__main__":
    run_erf_probe()
