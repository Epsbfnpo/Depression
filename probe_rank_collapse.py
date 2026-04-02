import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 导入原有项目模块
from models import DepMamba
from datasets import get_dvlog_dataloader, get_lmvd_dataloader


def compute_effective_rank(features: torch.Tensor) -> float:
    """
    计算特征矩阵的有效秩 (Effective Rank)
    :param features: 形状为 (N, D) 的特征矩阵，N 为样本数，D 为特征维度
    """
    # 1. 均值中心化
    features_centered = features - features.mean(dim=0, keepdim=True)

    # 2. 对特征矩阵进行 SVD 分解
    # features_centered 的奇异值平方正比于其协方差矩阵的特征值
    U, S, Vh = torch.linalg.svd(features_centered, full_matrices=False)
    eigenvalues = S ** 2

    # 3. 计算归一化概率分布 p_i
    p = eigenvalues / eigenvalues.sum()

    # 4. 过滤极小值以避免 log(0) 错误
    p = p[p > 1e-10]

    # 5. 计算奇异值香农熵 (SVSE) 并求指数得到有效秩
    svse = -torch.sum(p * torch.log(p))
    effective_rank = torch.exp(svse).item()

    return effective_rank


def main():
    # --- 1. 加载配置与模型设置 (复用原有 Config) ---
    config_path = "./config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", ["cuda"])[0] if torch.cuda.is_available() else "cpu")
    dataset_name = config.get("dataset", "dvlog")
    batch_size = config.get("batch_size", 16)
    data_dir = os.path.join(config.get("data_dir"), dataset_name)
    test_gender = config.get("test_gender", "both")

    print(f"[*] 初始化探针: 目标数据集 -> {dataset_name.upper()}")

    # --- 2. 初始化模型并加载最佳权重 ---
    if dataset_name == 'lmvd':
        net = DepMamba(**config['mmmamba_lmvd'])
        val_loader = get_lmvd_dataloader(data_dir, "valid", batch_size, test_gender, aug=False)
    else:
        net = DepMamba(**config['mmmamba'])
        val_loader = get_dvlog_dataloader(data_dir, "valid", batch_size, test_gender, aug=False)

    net = net.to(device)

    # 寻找最佳权重文件 (假设取第0次迭代的权重，你可以根据需要修改下标)
    iter_idx = 0
    weight_path = f"{config['save_dir']}/{dataset_name}_DepMamba_{iter_idx}/checkpoints/best_model.pt"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件，请确认路径: {weight_path}")

    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()
    print(f"[*] 成功加载模型权重: {weight_path}")

    # --- 3. 全量推理并提取深层特征 ---
    Za_list, Zv_list, Zf_list = [], [], []

    print("[*] 正在验证集上执行前向传播并捕获空间特征...")
    with torch.no_grad():
        for x, y, mask in tqdm(val_loader, desc="Probing Space"):
            x = x.to(device)
            mask = mask.to(device)

            # 处理 DataParallel 包装
            base_net = net.module if hasattr(net, 'module') else net

            # === 解构特征提取流程 ===
            xa = x[:, :, 136:]
            xv = x[:, :, :136]

            # 初始 1x1 卷积映射
            xa = base_net.conv_audio(xa.permute(0, 2, 1)).permute(0, 2, 1)
            xv = base_net.conv_video(xv.permute(0, 2, 1)).permute(0, 2, 1)

            # 经过多模态协作 SSM (CoSSM)
            xa_out, xv_out = base_net.cossm_encoder(xa, xv)

            # 经过多模态增强 SSM (EnSSM)
            x_cat = torch.cat([xa_out, xv_out], dim=-1)
            x_fused = base_net.enssm_encoder(x_cat)

            # === 时间维度池化 (严格对齐原 DepMamba 代码) ===
            mask_float = mask.unsqueeze(-1).float()

            # 单模态 Audio 特征 (Za)
            za = xa_out * mask_float
            za = za.sum(dim=1) / mask_float.sum(dim=1, keepdim=False)

            # 单模态 Video 特征 (Zv)
            zv = xv_out * mask_float
            zv = zv.sum(dim=1) / mask_float.sum(dim=1, keepdim=False)

            # 融合后特征 (Zf)
            zf = x_fused * mask_float
            zf = zf.sum(dim=1) / mask_float.sum(dim=1, keepdim=False)

            Za_list.append(za)
            Zv_list.append(zv)
            Zf_list.append(zf)

    # 聚合全量验证集特征
    Za_tensor = torch.cat(Za_list, dim=0)  # 形状: (N, 256)
    Zv_tensor = torch.cat(Zv_list, dim=0)  # 形状: (N, 256)
    Zf_tensor = torch.cat(Zf_list, dim=0)  # 形状: (N, 512)

    # --- 4. 探针计算与病理诊断 ---
    er_a = compute_effective_rank(Za_tensor)
    er_v = compute_effective_rank(Zv_tensor)
    er_f = compute_effective_rank(Zf_tensor)

    max_single_er = max(er_a, er_v)

    print("\n" + "=" * 50)
    print(f"📊 多模态融合几何陷阱探针结果")
    print("=" * 50)
    print(f"🔊 音频特征 (Z_a) 理论维度: {Za_tensor.shape[1]} | 有效秩 ER: {er_a:.2f}")
    print(f"👁️ 视频特征 (Z_v) 理论维度: {Zv_tensor.shape[1]} | 有效秩 ER: {er_v:.2f}")
    print(f"🌀 融合特征 (Z_f) 理论维度: {Zf_tensor.shape[1]} | 有效秩 ER: {er_f:.2f}")
    print("-" * 50)
    print(f"📈 诊断阈值 (max(ER_a, ER_v)): {max_single_er:.2f}")

    if er_f <= max_single_er:
        print("\n🚨 [病理确诊]：实锤发生“特征秩坍缩” (Rank Collapse)！")
        print("诊断说明：虽然你在物理维度上通过 concat 将维度提升到了 512，但有效的信息维度甚至不敌单模态最高峰。这意味着 EnSSM 并没有引发 Synergy (协同)，而是在过度拟合冗余的共享子空间，建议考虑解除 Shared Transition Matrix 或增加正交约束损失 (Orthogonal Regularization)。")
    elif er_f < (er_a + er_v) * 0.7:
        print("\n⚠️ [轻度病理]：存在明显的模态表征挤压现象。")
        print("诊断说明：虽然融合特征的有效秩超过了单模态，但显著低于两者的理论叠加。融合空间中包含了大量高度相关的同质化信息。")
    else:
        print("\n✅ [健康]：表征空间健康，多模态特征实现了有效的信息正交与互补。")
    print("=" * 50)


if __name__ == "__main__":
    main()
