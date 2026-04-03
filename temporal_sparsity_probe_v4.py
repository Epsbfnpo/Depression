import os

import matplotlib.pyplot as plt
import torch
import yaml

from datasets.dvlog import get_dvlog_dataloader
from models import DepMamba


def load_config(config_path: str = "./config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_single_tensor(model: DepMamba, tensor: torch.Tensor, device: torch.device) -> float:
    """给定单个 [T, D] 特征张量，输出抑郁概率。"""
    L = tensor.shape[0]
    x = tensor.unsqueeze(0).to(device)
    mask = torch.ones((1, L), dtype=torch.long, device=device)
    logits = model(x, mask)
    return torch.sigmoid(logits).item()


def find_best_anchors(model: DepMamba, dataloader, device: torch.device):
    """步骤 1：遍历数据集，自动寻找“最纯”抑郁锚点与健康锚点。"""
    print("\n[阶段 1: 扫描数据集，寻找完美锚点 (Anchor)...]")
    best_dep_tensor, best_nor_tensor = None, None
    max_dep_prob, min_nor_prob = 0.0, 1.0

    for features, labels, padding_mask in dataloader:
        valid_len = int(padding_mask[0].sum().item())
        feature = features[0][:valid_len]
        label = int(labels[0].item())

        prob = predict_single_tensor(model, feature, device)
        if label == 1 and prob > max_dep_prob:
            max_dep_prob = prob
            best_dep_tensor = feature
        elif label == 0 and prob < min_nor_prob:
            min_nor_prob = prob
            best_nor_tensor = feature

        if max_dep_prob > 0.90 and min_nor_prob < 0.10:
            break

    if best_dep_tensor is None or best_nor_tensor is None:
        raise RuntimeError("未找到可用锚点：请检查 checkpoint 或数据集标签。")

    print(
        f"-> 锁定抑郁锚点: 原始预测概率 {max_dep_prob * 100:.2f}%, 长度 {best_dep_tensor.shape[0]}"
    )
    print(
        f"-> 锁定健康锚点: 原始预测概率 {min_nor_prob * 100:.2f}%, 长度 {best_nor_tensor.shape[0]}"
    )
    return best_dep_tensor, best_nor_tensor


def extract_peak_pathology(
    model: DepMamba,
    dep_tensor: torch.Tensor,
    device: torch.device,
    window_size: int = 150,
    stride: int = 30,
) -> torch.Tensor:
    """步骤 2：滑动窗口寻找抑郁样本中的高浓缩病灶片段。"""
    print(f"\n[阶段 2: 在抑郁锚点中提取高浓缩病灶 (窗口大小={window_size})...]")
    total_len = dep_tensor.shape[0]
    if total_len <= window_size:
        print("-> 抑郁锚点长度不超过窗口，直接使用整段。")
        return dep_tensor

    best_prob = -1.0
    best_start = 0
    for start in range(0, total_len - window_size + 1, stride):
        clip = dep_tensor[start : start + window_size]
        prob = predict_single_tensor(model, clip, device)
        if prob > best_prob:
            best_prob = prob
            best_start = start

    peak_clip = dep_tensor[best_start : best_start + window_size]
    print(
        f"-> 成功提取片段 [{best_start}:{best_start + window_size}],"
        f" 纯片段预测概率: {best_prob * 100:.2f}%"
    )
    return peak_clip


def tensor_cyber_surgery(healthy_tensor: torch.Tensor, peak_clip: torch.Tensor, total_L: int) -> torch.Tensor:
    """步骤 3：将病灶固定植入健康背景中央。"""
    dep_length = peak_clip.shape[0]
    pad_length = total_L - dep_length
    if pad_length <= 0:
        return peak_clip

    front_pad_len = pad_length // 2
    back_pad_len = pad_length - front_pad_len

    while healthy_tensor.shape[0] < max(front_pad_len, back_pad_len):
        healthy_tensor = torch.cat([healthy_tensor, healthy_tensor], dim=0)

    healthy_front = healthy_tensor[:front_pad_len]
    healthy_back = healthy_tensor[-back_pad_len:]
    return torch.cat([healthy_front, peak_clip, healthy_back], dim=0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    model = DepMamba(**config["mmmamba"]).to(device)

    checkpoint_path = (
        "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DepMamba/"
        "mambamodels/dvlog_DepMamba_2/checkpoints/best_model.pt"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 使用原项目 dataloader，确保特征读取与训练/测试流程一致
    data_root = "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/dvlog"
    test_loader = get_dvlog_dataloader(root=data_root, fold="test", batch_size=1, aug=False)

    with torch.no_grad():
        dep_anchor, nor_anchor = find_best_anchors(model, test_loader, device)
        peak_clip = extract_peak_pathology(model, dep_anchor, device, window_size=150, stride=30)

        print("\n[阶段 3: 执行时序稀疏性与时长衰减探针实验...]")
        L_list = [150, 200, 300, 500, 800, 1200, 2000]
        probabilities = []

        for L in L_list:
            test_tensor = tensor_cyber_surgery(nor_anchor, peak_clip, total_L=L)
            prob = predict_single_tensor(model, test_tensor, device)
            probabilities.append(prob)
            print(f"背景拉长至 L={L:<5d} (病灶占比 {150 / L * 100:.1f}%) | 抑郁预测概率: {prob * 100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(L_list, probabilities, marker="o", linewidth=2, color="b")
    plt.axhline(y=0.5, color="r", linestyle="--", label="Decision Threshold (0.5)")
    for i, L in enumerate(L_list):
        plt.annotate(
            f"{probabilities[i] * 100:.1f}%",
            (L_list[i], probabilities[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.title("Probe 3: Temporal Sparsity & State Dilution in Mamba", fontsize=14)
    plt.xlabel("Total Sequence Length L (Frames)", fontsize=12)
    plt.ylabel("Predicted Probability of Depression", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs("./probe_results", exist_ok=True)
    plt.savefig("./probe_results/temporal_sparsity_curve_final.png")
    print("\n探针实验完成！衰减曲线已保存至 ./probe_results/temporal_sparsity_curve_final.png")


if __name__ == "__main__":
    main()
