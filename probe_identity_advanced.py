import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from models.DepMamba import DepMamba
from datasets import get_dvlog_dataloader, get_lmvd_dataloader


def load_config(config_path="./config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = load_config()
    device = torch.device(config.get("device", ["cuda"])[0] if torch.cuda.is_available() else "cpu")

    # 1. 载入模型配置
    dataset_name = config.get("dataset", "dvlog")
    if dataset_name == 'lmvd':
        model_config = config['mmmamba_lmvd']
    else:
        model_config = config['mmmamba']

    model = DepMamba(**model_config)

    # 请确保路径与你实际的权重路径一致
    checkpoint_path = f"/datasets/work/hb-nhmrc-dhcp/work/liu275/Depression/DepMamba/mambamodels/dvlog_DepMamba_2/checkpoints/best_model.pt"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO] 成功加载模型权重: {checkpoint_path}")
    else:
        print(f"[WARN] 未找到模型权重 {checkpoint_path}，请确保你已经训练过模型！")
        return

    model.to(device)
    model.eval()

    # 2. 准备测试数据
    data_dir = os.path.join(config['data_dir'], dataset_name)
    if dataset_name == 'dvlog':
        test_loader = get_dvlog_dataloader(data_dir, "test", batch_size=1, gender=config.get('test_gender', 'both'), aug=False)
    elif dataset_name == 'lmvd':
        test_loader = get_lmvd_dataloader(data_dir, "test", batch_size=1, gender=config.get('test_gender', 'both'), aug=False)
    else:
        raise ValueError("不支持的数据集")

    print("\n" + "=" * 60)
    print("🚀 开始执行：多模态交叉消融探针 (Multimodal Ablation Probe)")
    print("=" * 60)

    # 统计器
    total_depressed = 0
    original_correct = 0

    # 探针计数器：在不同干预下，模型【依然预测为抑郁】的样本数
    count_frozenV_normalA = 0  # 探针1：视觉冻结 + 音频正常 (你的初版)
    count_frozenV_zeroA = 0    # 探针2：视觉冻结 + 音频置零 (终极身份过拟合测试)
    count_zeroV_normalA = 0    # 探针3：视觉置零 + 音频正常 (纯听觉依赖测试)
    count_normalV_zeroA = 0    # 探针4：视觉正常 + 音频置零 (纯视觉动态测试)

    with torch.no_grad():
        for x, y, mask in tqdm(test_loader, desc="Probing"):
            if y.item() != 1:  # 只测真实的抑郁症样本
                continue

            total_depressed += 1
            x = x.to(device)
            mask = mask.to(device)
            seq_len = x.shape[1]

            # --- Baseline: 原始预测 ---
            y_pred_orig = model(x, mask)
            if (y_pred_orig > 0.).item():
                original_correct += 1

                # 提取第一帧用于冻结视觉
                first_frame_visual = x[:, 0:1, :136]

                # ==========================================
                # 🔪 探针 1：视觉冻结 + 音频正常
                # 测试：当人脸变成面具时，音频能提供多大的兜底能力？
                # ==========================================
                x_p1 = x.clone()
                x_p1[:, :, :136] = first_frame_visual.expand(-1, seq_len, -1)
                if (model(x_p1, mask) > 0.).item():
                    count_frozenV_normalA += 1

                # ==========================================
                # 🔪 探针 2：视觉冻结 + 音频置零（关键修复！）
                # 测试：极致的受试者身份过拟合！没声音且脸不动，还能认出来吗？
                # ==========================================
                x_p2 = x.clone()
                x_p2[:, :, :136] = first_frame_visual.expand(-1, seq_len, -1)
                x_p2[:, :, 136:] = 0.0  # 音频彻底静音
                if (model(x_p2, mask) > 0.).item():
                    count_frozenV_zeroA += 1

                # ==========================================
                # 🔪 探针 3：视觉完全置零 + 音频正常
                # 测试：蒙上模型的眼睛，只听声音，能看病吗？
                # ==========================================
                x_p3 = x.clone()
                x_p3[:, :, :136] = 0.0  # 视觉致盲
                if (model(x_p3, mask) > 0.).item():
                    count_zeroV_normalA += 1

                # ==========================================
                # 🔪 探针 4：视觉正常 + 音频完全置零
                # 测试：堵上模型的耳朵，只看动态脸，能看病吗？
                # ==========================================
                x_p4 = x.clone()
                x_p4[:, :, 136:] = 0.0  # 听觉致聋
                if (model(x_p4, mask) > 0.).item():
                    count_normalV_zeroA += 1

    # --- 报告生成 ---
    print("\n\n" + "📊" * 30)
    print(" " * 15 + "多模态探针诊断报告")
    print("📊" * 30)
    print(f"📌 测试集抑郁症样本总数: {total_depressed}")
    print(f"✅ 基线(Baseline) 正确识别数: {original_correct} (作为后续百分比的分母)")
    print("-" * 60)

    if original_correct > 0:
        rate_p1 = (count_frozenV_normalA / original_correct) * 100
        rate_p2 = (count_frozenV_zeroA / original_correct) * 100
        rate_p3 = (count_zeroV_normalA / original_correct) * 100
        rate_p4 = (count_normalV_zeroA / original_correct) * 100

        print(f"🔪 探针1 [面具脸 + 正常说话]: {count_frozenV_normalA} 例保持预测 ({rate_p1:.2f}%)")
        print(f"🔪 探针2 [面具脸 + 彻底静音]: {count_frozenV_zeroA} 例保持预测 ({rate_p2:.2f}%)   <-- 核心过拟合指标")
        print(f"🔪 探针3 [致盲   + 正常说话]: {count_zeroV_normalA} 例保持预测 ({rate_p3:.2f}%)")
        print(f"🔪 探针4 [正常脸 + 彻底静音]: {count_normalV_zeroA} 例保持预测 ({rate_p4:.2f}%)")
        print("-" * 60)

        # 智能诊断结论
        print("💡 【自动化病理诊断分析】")

        # 1. 分析身份过拟合
        if rate_p2 > 60:
            print("🚨 结论 A: [灾难性视觉身份过拟合]")
            print(f"在切断音频且冻结面部的情况下，仍有 {rate_p2:.2f}% 的样本被判为抑郁。实锤模型在视觉上是个'人脸识别器'，严重依赖静态骨骼特征而非动态病理。")
        else:
            print("✅ 结论 A: [视觉身份特征解耦良好]")
            print(f"切断音频并冻结面部后，置信度降至 {rate_p2:.2f}%。模型沉冤得雪！它并没有死记硬背患者长相，你的初版探针高分确实是因为音频泄露。")

        # 2. 分析模态依赖度
        if rate_p3 > rate_p4:
            print("🎧 结论 B: [模态偏好 - 听觉主导]")
            print(f"致盲测试({rate_p3:.2f}%) 分数高于 聋哑测试({rate_p4:.2f}%)，说明当前模型在诊断中更依赖【音频模态】。患者的语音特征(迟缓/音调)是破局关键。")
        else:
            print("👁️ 结论 B: [模态偏好 - 视觉主导]")
            print(f"聋哑测试({rate_p4:.2f}%) 分数高于 致盲测试({rate_p3:.2f}%)，说明当前模型更依赖【视觉模态】。")

    else:
        print("模型未能正确识别任何原始样本，无法进行探针分析。")


if __name__ == '__main__':
    main()
