import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models import DepMamba


def load_config(config_path: str = "./config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_features(tensor: torch.Tensor) -> torch.Tensor:
    """
    【关键修复】特征标准化。
    强烈建议：如果训练时使用了全局 mean/std，请替换为全局归一化。
    这里先使用样本级 Z-score，避免长序列下的数值漂移和状态爆炸。
    """
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True) + 1e-8
    return (tensor - mean) / std


def tensor_cyber_surgery(
    healthy_tensor: torch.Tensor,
    depressed_tensor: torch.Tensor,
    total_L: int,
    dep_start_idx: int,
    dep_length: int = 30,
) -> torch.Tensor:
    # 取前 1/3 位置附近，减少“盲切到中性段”的概率。
    start_d = depressed_tensor.shape[0] // 3
    dep_clip = depressed_tensor[start_d : start_d + dep_length, :]

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
    """辅助函数：输入单个 [T, D] 张量并输出预测概率。"""
    L = tensor.shape[0]
    x = tensor.unsqueeze(0).to(device)
    mask = torch.ones((1, L), dtype=torch.long, device=device)
    logits = model(x, mask)
    return torch.sigmoid(logits).item()


def main() -> None:
    # ================= 1. 初始化设置与模型加载 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    mamba_cfg = config["mmmamba"]
    model = DepMamba(**mamba_cfg).to(device)

    checkpoint_path = (
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DepMamba/"
        "mambamodels/dvlog_DepMamba_2/checkpoints/best_model.pt"
    )
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"成功加载权重: {checkpoint_path}")
    else:
        raise FileNotFoundError("必须使用真实权重进行探针实验！")

    model.eval()

    # ================= 2. 准备与【归一化】探针数据 =================
    print("\n[加载并处理真实数据...]")
    dep_v = np.load(
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/"
        "dvlog/42/42_visual.npy"
    )
    dep_a = np.load(
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/"
        "dvlog/42/42_acoustic.npy"
    )
    dep_np = np.concatenate((dep_v, dep_a), axis=-1)
    depressed_tensor = torch.from_numpy(dep_np).float()

    nor_v = np.load(
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/"
        "dvlog/960/960_visual.npy"
    )
    nor_a = np.load(
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/"
        "dvlog/960/960_acoustic.npy"
    )
    nor_np = np.concatenate((nor_v, nor_a), axis=-1)
    healthy_tensor = torch.from_numpy(nor_np).float()

    depressed_tensor = normalize_features(depressed_tensor)
    healthy_tensor = normalize_features(healthy_tensor)

    repeat_times = 8000 // healthy_tensor.shape[0] + 1
    healthy_tensor = healthy_tensor.repeat(repeat_times, 1)

    # ================= 3. 基准校验 (Sanity Check) =================
    print("\n[开始基准校验 (Sanity Check) - 验证模型对未经剪裁样本的判断力]")
    with torch.no_grad():
        prob_dep = predict_single_tensor(model, depressed_tensor, device)
        prob_nor = predict_single_tensor(model, healthy_tensor[:1500, :], device)

        print(f"完整样本 42 (抑郁 Label=1) 预测概率: {prob_dep * 100:.2f}%")
        print(f"完整样本 960 (健康 Label=0) 预测概率: {prob_nor * 100:.2f}%")

        if prob_dep < 0.5 or prob_nor > 0.5:
            print("\n⚠️ 警告：模型在完整样本上的预测已失败，探针结果可信度较低。")
            print("请重点核对：")
            print("1. 模型 checkpoint 对应的测试性能是否稳定；")
            print("2. 当前 normalize_features 是否与训练阶段处理一致。")

    # ================= 4. 运行探针实验 =================
    dep_length = 30
    L_list = [30, 40, 50, 75, 100, 150, 200, 500, 1000, 2000, 5000]
    probabilities = []

    print("\n[开始执行时序稀疏性探针实验 (Temporal Sparsity Probe)...]")
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

            prob = predict_single_tensor(model, test_tensor, device)
            probabilities.append(prob)
            print(f"序列总长度 L={L:<5d} | 预测为抑郁症的概率: {prob * 100:.2f}%")

    # ================= 5. 绘制探针衰减曲线 =================
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
