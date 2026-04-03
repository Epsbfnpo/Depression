import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from datasets.dvlog import get_dvlog_dataloader
from models import DepMamba


def load_config(config_path: str = "./config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ================= 工具函数 =================
def predict_single_tensor(model: DepMamba, tensor: torch.Tensor, device: torch.device) -> float:
    L = tensor.shape[0]
    x = tensor.unsqueeze(0).to(device)
    mask = torch.ones((1, L), dtype=torch.long, device=device)
    logits = model(x, mask)
    return torch.sigmoid(logits).item()


def find_best_anchors(model: DepMamba, dataloader, device: torch.device):
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

    return best_dep_tensor, best_nor_tensor


def extract_peak_pathology(
    model: DepMamba,
    dep_tensor: torch.Tensor,
    device: torch.device,
    window_size: int = 150,
) -> torch.Tensor:
    L = dep_tensor.shape[0]
    if L <= window_size:
        return dep_tensor

    best_prob, best_start = 0.0, 0
    for start in range(0, L - window_size + 1, 30):
        clip = dep_tensor[start : start + window_size]
        prob = predict_single_tensor(model, clip, device)
        if prob > best_prob:
            best_prob = prob
            best_start = start

    return dep_tensor[best_start : best_start + window_size]


def tensor_cyber_surgery(healthy_tensor: torch.Tensor, peak_clip: torch.Tensor, total_L: int) -> torch.Tensor:
    dep_length = peak_clip.shape[0]
    pad_length = total_L - dep_length
    if pad_length <= 0:
        return peak_clip

    front_pad_len = pad_length // 2
    back_pad_len = pad_length - front_pad_len

    while healthy_tensor.shape[0] < max(front_pad_len, back_pad_len):
        healthy_tensor = torch.cat([healthy_tensor, healthy_tensor], dim=0)

    return torch.cat(
        [healthy_tensor[:front_pad_len], peak_clip, healthy_tensor[-back_pad_len:]], dim=0
    )


# ================= 核心：Hook 捕获器 =================
intermediate_features = {}


def get_features(name: str):
    def hook(_module, _input, output):
        # 截取 enssm_encoder 的输出: [Batch, L, Dim]
        intermediate_features[name] = output.detach()

    return hook


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    model = DepMamba(**config["mmmamba"]).to(device)

    checkpoint_path = "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DepMamba/mambamodels/dvlog_DepMamba_2/checkpoints/best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 注册 Forward Hook 到 enssm_encoder 上
    model.enssm_encoder.register_forward_hook(get_features("enssm_out"))

    data_root = "/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DVLOG_Feature/dvlog"
    test_loader = get_dvlog_dataloader(root=data_root, fold="test", batch_size=1, aug=False)

    with torch.no_grad():
        print("[阶段 1] 寻找锚点与提取病灶...")
        dep_anchor, nor_anchor = find_best_anchors(model, test_loader, device)
        peak_clip = extract_peak_pathology(model, dep_anchor, device, window_size=150)

        print("\n[阶段 2] 执行池化前激活探针 (Pre-pooling Activation Probe)...")
        L_list = [150, 500, 2000]
        activation_curves = {}

        for L in L_list:
            test_tensor = tensor_cyber_surgery(nor_anchor, peak_clip, total_L=L)

            # 前向传播，触发 hook 捕获特征
            prob = predict_single_tensor(model, test_tensor, device)

            # 取出捕获的 [1, L, D] 特征
            hidden_states = intermediate_features["enssm_out"].squeeze(0)  # [L, D]

            # 计算每帧的 L2 Norm (激活性强度)
            frame_activations = hidden_states.norm(dim=-1).cpu().numpy()
            activation_curves[L] = frame_activations
            print(f"-> 测试长度 L={L:<5d} | 预测概率: {prob*100:.2f}% | 特征图已捕获")

    # ================= 画图：激活强度对齐对比 =================
    plt.figure(figsize=(12, 6))

    # 颜色深浅代表序列长度
    colors = {150: "red", 500: "orange", 2000: "blue"}

    for L, curve in activation_curves.items():
        # 为了公平对比，我们把 X 轴对齐到病灶区域
        # 病灶总是插在中间，所以起点是 (L - 150) / 2
        start_idx = (L - 150) // 2

        # 将 X 坐标平移，使得 0 对应病灶的第一帧
        x_axis = np.arange(L) - start_idx

        # 绘制激活曲线
        plt.plot(x_axis, curve, label=f"Total L={L}", color=colors[L], alpha=0.8, linewidth=1.5)

    # 标记出病灶区 (Lesion Window)
    plt.axvspan(0, 150, color="red", alpha=0.1, label="Injected Lesion Window (150 frames)")

    plt.title("Pre-pooling Activation Map: Does Mamba Forget or Does Pooling Dilute?", fontsize=14)
    plt.xlabel("Frame Index (Relative to Lesion Start)", fontsize=12)
    plt.ylabel("Feature Activation Intensity (L2 Norm)", fontsize=12)
    plt.xlim(-200, 350)
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs("./probe_results", exist_ok=True)
    output_path = "./probe_results/pre_pooling_activation.png"
    plt.savefig(output_path)
    print(f"\n探针实验完成！神级诊断图已保存至 {output_path}")


if __name__ == "__main__":
    main()
