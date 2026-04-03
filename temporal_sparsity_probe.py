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
    dep_length: int = 150,
) -> torch.Tensor:
    """截取病理片段并嵌入健康背景。"""
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
    return torch.cat([healthy_front, dep_clip, healthy_back], dim=0)


def predict_single_tensor(model: DepMamba, tensor: torch.Tensor, device: torch.device) -> float:
    """输入 [T, D] 特征张量，输出抑郁概率。"""
    L = tensor.shape[0]
    x = tensor.unsqueeze(0).to(device)
    mask = torch.ones((1, L), dtype=torch.long, device=device)
    logits = model(x, mask)
    return torch.sigmoid(logits).item()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    mamba_cfg = config["mmmamba"]
    model = DepMamba(**mamba_cfg).to(device)

    checkpoint_path = (
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DepMamba/"
        "mambamodels/dvlog_DepMamba_2/checkpoints/best_model.pt"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("\n[加载 RAW 数据 (与 dvlog.py 完全对齐)...]")
    # 抑郁样本
    dep_v = np.load(
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/"
        "dvlog/42/42_visual.npy"
    )
    dep_a = np.load(
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/"
        "dvlog/42/42_acoustic.npy"
    )
    T_dep = min(dep_v.shape[0], dep_a.shape[0])
    dep_np = np.concatenate((dep_v[:T_dep], dep_a[:T_dep]), axis=1).astype(np.float32)
    depressed_tensor = torch.from_numpy(dep_np)

    # 健康样本
    nor_v = np.load(
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/"
        "dvlog/960/960_visual.npy"
    )
    nor_a = np.load(
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/"
        "dvlog/960/960_acoustic.npy"
    )
    T_nor = min(nor_v.shape[0], nor_a.shape[0])
    nor_np = np.concatenate((nor_v[:T_nor], nor_a[:T_nor]), axis=1).astype(np.float32)
    healthy_tensor = torch.from_numpy(nor_np)

    print("\n[第一步：基准校验 (Sanity Check) - 原版张量直接推理]")
    with torch.no_grad():
        prob_dep = predict_single_tensor(model, depressed_tensor, device)
        prob_nor = predict_single_tensor(model, healthy_tensor, device)
        print(f"-> 完整样本 42 (抑郁 Label=1, 原长 {T_dep}) 预测概率: {prob_dep * 100:.2f}%")
        print(f"-> 完整样本 960 (健康 Label=0, 原长 {T_nor}) 预测概率: {prob_nor * 100:.2f}%")

    repeat_times = 8000 // healthy_tensor.shape[0] + 1
    healthy_tensor_long = healthy_tensor.repeat(repeat_times, 1)

    print("\n[第二步：验证“长度偏见”假设]")
    with torch.no_grad():
        prob_long_healthy = predict_single_tensor(model, healthy_tensor_long[:5000], device)
        print(f"-> 纯健康样本暴力延长到 5000 帧 (Label=0) 预测概率: {prob_long_healthy * 100:.2f}%")
        if prob_long_healthy > 0.5:
            print("   ⚠️ 确诊：模型存在【序列长度偏见】，超长输入可能导致状态漂移。")

    print("\n[第三步：执行时序稀疏性探针实验 (插入 150 帧病理特征)...]")
    dep_length = 150
    L_list = [150, 200, 300, 500, 1000, 2000, 5000]
    probabilities = []

    with torch.no_grad():
        for L in L_list:
            insert_idx = (L - dep_length) // 2
            test_tensor = tensor_cyber_surgery(
                healthy_tensor_long,
                depressed_tensor,
                total_L=L,
                dep_start_idx=insert_idx,
                dep_length=dep_length,
            )
            prob = predict_single_tensor(model, test_tensor, device)
            probabilities.append(prob)
            print(f"序列总长度 L={L:<5d} | 预测为抑郁症的概率: {prob * 100:.2f}%")

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
    plt.title("Temporal Sparsity Probe V3: DepMamba Survival Curve", fontsize=14)
    plt.xlabel("Total Sequence Length L (Frames)", fontsize=12)
    plt.ylabel("Predicted Probability of Depression", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs("./probe_results", exist_ok=True)
    plt.savefig("./probe_results/temporal_sparsity_curve_v3.png")
    print("\n探针实验完成！衰减曲线已保存至 ./probe_results/temporal_sparsity_curve_v3.png")


if __name__ == "__main__":
    main()
