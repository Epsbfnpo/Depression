import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models import DepMamba


def load_config(config_path: str = "./config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tensor_cyber_surgery(
    healthy_tensor: torch.Tensor,
    depressed_tensor: torch.Tensor,
    total_L: int,
    dep_start_idx: int,
    dep_length: int = 30,
) -> torch.Tensor:
    """
    赛博外科手术：将短促的抑郁特征强行植入健康背景张量中。

    Args:
        healthy_tensor: [T_h, Dim] 健康张量。
        depressed_tensor: [T_d, Dim] 抑郁张量。
        total_L: 目标总序列长度。
        dep_start_idx: 抑郁片段插入的起始位置。
        dep_length: 截取的抑郁片段长度（默认30帧，约1秒）。
    """
    mid_d = depressed_tensor.shape[0] // 2
    dep_clip = depressed_tensor[mid_d : mid_d + dep_length, :]

    pad_length = total_L - dep_length
    if pad_length <= 0:
        return dep_clip

    front_pad_len = dep_start_idx
    back_pad_len = pad_length - front_pad_len

    healthy_front = healthy_tensor[:front_pad_len, :]
    healthy_back = (
        healthy_tensor[-back_pad_len:, :]
        if back_pad_len > 0
        else torch.empty((0, healthy_tensor.shape[1]))
    )

    fused_tensor = torch.cat([healthy_front, dep_clip, healthy_back], dim=0)
    return fused_tensor


def main() -> None:
    # ================= 1. 初始化设置与模型加载 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    dataset_type = "dvlog"  # 或 "lmvd"
    mamba_cfg = config["mmmamba"] if dataset_type == "dvlog" else config["mmmamba_lmvd"]

    model = DepMamba(**mamba_cfg).to(device)

    checkpoint_path = (
        f"{config['save_dir']}/{dataset_type}_DepMamba_0/checkpoints/best_model.pt"
    )
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"成功加载权重: {checkpoint_path}")
    else:
        print("警告: 未找到权重，使用随机初始化进行演示。")

    model.eval()

    # ================= 2. 准备真实探针数据 =================
    # 请将下面的路径替换为您实际的 .npy 文件路径！
    # 1. 加载抑郁症样本 (提取病理微表情)
    dep_v = np.load("/您的路径/dep_v.npy")
    dep_a = np.load("/您的路径/dep_a.npy")
    dep_np = np.concatenate((dep_v, dep_a), axis=-1)
    depressed_tensor = torch.from_numpy(dep_np).float()

    # 2. 加载健康样本 (作为漫长的背景)
    nor_v = np.load("/您的路径/nor_v.npy")
    nor_a = np.load("/您的路径/nor_a.npy")
    nor_np = np.concatenate((nor_v, nor_a), axis=-1)

    # 真实健康视频可能不够长，循环复制到足够长度以模拟长背景。
    healthy_tensor = torch.from_numpy(nor_np).float()
    repeat_times = 8000 // healthy_tensor.shape[0] + 1
    healthy_tensor = healthy_tensor.repeat(repeat_times, 1)

    dep_length = 30
    L_list = [30, 40, 50, 75, 100, 150, 200, 500, 1000, 2000, 5000]
    probabilities = []

    # ================= 3. 运行探针实验 =================
    print("\n开始执行时序稀疏性探针实验 (Temporal Sparsity Probe)...")
    with torch.no_grad():
        for L in L_list:
            insert_idx = (L - dep_length) // 2

            test_tensor = tensor_cyber_surgery(
                healthy_tensor,
                depressed_tensor,
                total_L=L,
                dep_start_idx=insert_idx,
                dep_length=dep_length,
            )

            x = test_tensor.unsqueeze(0).to(device)
            mask = torch.ones((1, L)).long().to(device)

            logits = model(x, mask)
            prob = torch.sigmoid(logits).item()
            probabilities.append(prob)

            print(f"序列总长度 L={L:<5d} | 预测为抑郁症的概率: {prob * 100:.2f}%")

    # ================= 4. 绘制探针衰减曲线 =================
    plt.figure(figsize=(10, 6))
    plt.plot(L_list, probabilities, marker="o", linewidth=2, color="b")
    plt.axhline(y=0.5, color="r", linestyle="--", label="Decision Threshold (0.5)")

    for i, L in enumerate(L_list):
        ratio = dep_length / L * 100
        plt.annotate(
            f"Ratio:{ratio:.1f}%",
            (L_list[i], probabilities[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.title("Temporal Sparsity Probe: DepMamba Survival Curve", fontsize=14)
    plt.xlabel("Total Sequence Length L (Frames)", fontsize=12)
    plt.ylabel("Predicted Probability of Depression", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs("./probe_results", exist_ok=True)
    plt.savefig("./probe_results/temporal_sparsity_curve.png")
    print("\n探针实验完成！衰减曲线已保存至 ./probe_results/temporal_sparsity_curve.png")


if __name__ == "__main__":
    main()
