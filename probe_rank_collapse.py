import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models import DepMamba
from datasets import get_dvlog_dataloader, get_lmvd_dataloader

def compute_effective_rank(features: torch.Tensor) -> float:
    features_centered = features - features.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(features_centered, full_matrices=False)
    eigenvalues = S ** 2
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 1e-10]
    svse = -torch.sum(p * torch.log(p))
    effective_rank = torch.exp(svse).item()
    return effective_rank

def main():
    config_path = "./config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(config.get("device", ["cuda"])[0] if torch.cuda.is_available() else "cpu")
    dataset_name = config.get("dataset", "dvlog")
    batch_size = config.get("batch_size", 16)
    data_dir = os.path.join(config.get("data_dir"), dataset_name)
    test_gender = config.get("test_gender", "both")
    print(f"[*] 初始化探针: 目标数据集 -> {dataset_name.upper()}")
    if dataset_name == 'lmvd':
        net = DepMamba(**config['mmmamba_lmvd'])
        val_loader = get_lmvd_dataloader(data_dir, "valid", batch_size, test_gender, aug=False)
    else:
        net = DepMamba(**config['mmmamba'])
        val_loader = get_dvlog_dataloader(data_dir, "valid", batch_size, test_gender, aug=False)
    net = net.to(device)
    weight_path = f"/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DepMamba/mambamodels/dvlog_DepMamba_2/checkpoints/best_model.pt"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件，请确认路径: {weight_path}")
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()
    print(f"[*] 成功加载模型权重: {weight_path}")
    Za_list, Zv_list, Zf_list = [], [], []
    print("[*] 正在验证集上执行前向传播并捕获空间特征...")
    with torch.no_grad():
        for x, y, mask in tqdm(val_loader, desc="Probing Space"):
            x = x.to(device)
            mask = mask.to(device)
            base_net = net.module if hasattr(net, 'module') else net
            xa = x[:, :, 136:]
            xv = x[:, :, :136]
            xa = base_net.conv_audio(xa.permute(0, 2, 1)).permute(0, 2, 1)
            xv = base_net.conv_video(xv.permute(0, 2, 1)).permute(0, 2, 1)
            xa_out, xv_out = base_net.cossm_encoder(xa, xv)
            x_cat = torch.cat([xa_out, xv_out], dim=-1)
            x_fused = base_net.enssm_encoder(x_cat)
            mask_float = mask.unsqueeze(-1).float()
            za = xa_out * mask_float
            za = za.sum(dim=1) / mask_float.sum(dim=1, keepdim=False)
            zv = xv_out * mask_float
            zv = zv.sum(dim=1) / mask_float.sum(dim=1, keepdim=False)
            zf = x_fused * mask_float
            zf = zf.sum(dim=1) / mask_float.sum(dim=1, keepdim=False)
            Za_list.append(za)
            Zv_list.append(zv)
            Zf_list.append(zf)
    Za_tensor = torch.cat(Za_list, dim=0)
    Zv_tensor = torch.cat(Zv_list, dim=0)
    Zf_tensor = torch.cat(Zf_list, dim=0)
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
        print("诊断说明：虽然你在物理维度上通过 concat 将维度提升到了 512，但有效的信息维度甚至不敌单模态最高峰。这意味着 EnSSM 并没有引发 Synergy (协同)，而是在过度拟合冗余的共享子空间")
    elif er_f < (er_a + er_v) * 0.7:
        print("\n⚠️ [轻度病理]：存在明显的模态表征挤压现象。")
        print("诊断说明：虽然融合特征的有效秩超过了单模态，但显著低于两者的理论叠加。融合空间中包含了大量高度相关的同质化信息。")
    else:
        print("\n✅ [健康]：表征空间健康，多模态特征实现了有效的信息正交与互补。")
    print("=" * 50)

if __name__ == "__main__":
    main()
