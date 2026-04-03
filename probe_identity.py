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

    # 假设我们加载第0次实验跑出的 best_model (对应 main.py 中的保存逻辑)
    # 你可以根据实际情况修改这里的路径，例如使用 args 传入
    checkpoint_path = f"{config['save_dir']}/{dataset_name}_DepMamba_0/checkpoints/best_model.pt"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO] 成功加载模型权重: {checkpoint_path}")
    else:
        print(f"[WARN] 未找到模型权重 {checkpoint_path}，请确保你已经训练过模型！")
        return

    model.to(device)
    model.eval()

    # 2. 准备测试数据 (batch_size 设置为 1 方便逐个做手术)
    data_dir = os.path.join(config['data_dir'], dataset_name)
    if dataset_name == 'dvlog':
        test_loader = get_dvlog_dataloader(data_dir, "test", batch_size=1, gender=config.get('test_gender', 'both'), aug=False)
    elif dataset_name == 'lmvd':
        test_loader = get_lmvd_dataloader(data_dir, "test", batch_size=1, gender=config.get('test_gender', 'both'), aug=False)
    else:
        raise ValueError("不支持的数据集")

    print("\n" + "=" * 50)
    print("开始执行 维度三：身份信息纠缠探针实验 (Identity Entanglement Probe)")
    print("=" * 50)

    total_depressed = 0             # 数据集中的真实抑郁样本数
    original_correct = 0            # 原模型正确预测的样本数
    frozen_still_depressed = 0      # 冻结视觉后，依然被判为抑郁的样本数

    with torch.no_grad():
        for x, y, mask in tqdm(test_loader, desc="Probing"):
            # 只提取 ground truth 为抑郁症的样本 (y == 1)
            if y.item() != 1:
                continue

            total_depressed += 1
            x = x.to(device)
            mask = mask.to(device)

            # --- 步骤 A: 记录原始预测结果 ---
            # DepMamba 的输出是 logits，> 0 代表置信度 > 0.5
            y_pred_orig = model(x, mask)
            is_dep_orig = (y_pred_orig > 0.).item()

            if is_dep_orig:
                original_correct += 1

                # --- 步骤 B: 赛博外科手术 (抹除面部动态) ---
                # x 的 shape: [Batch=1, Seq_len, Feature_Dim]
                # 前 136 维是视觉坐标，后面是音频特征
                x_frozen = x.clone()
                seq_len = x.shape[1]

                # 提取第一帧的 136 维静态骨骼特征: shape [1, 1, 136]
                first_frame_visual = x_frozen[:, 0:1, :136]

                # 魔法代码：将第一帧的脸覆盖到时间轴上的每一帧，音频特征保留不变
                x_frozen[:, :, :136] = first_frame_visual.expand(-1, seq_len, -1)

                # --- 步骤 C: 测试冻结后的反应 ---
                y_pred_frozen = model(x_frozen, mask)
                is_dep_frozen = (y_pred_frozen > 0.).item()

                if is_dep_frozen:
                    frozen_still_depressed += 1

    # --- 步骤 D: 出具诊断报告 ---
    print("\n" + "=" * 50)
    print("🎯 探针诊断报告 🎯")
    print("=" * 50)
    print(f"1. 测试集中真实的强抑郁症样本总数: {total_depressed}")
    print(f"2. 模型在【正常动态下】正确识别出的抑郁症样本数: {original_correct}")

    if original_correct > 0:
        overfit_rate = (frozen_still_depressed / original_correct) * 100
        print(f"3. 【探针触发】在面部动态被完全清零的情况下，模型依然报警说他抑郁的样本数: {frozen_still_depressed}")
        print("-" * 50)
        print(f"👉 受试者身份过拟合率 (Identity Overfitting Rate): {overfit_rate:.2f}%")

        if overfit_rate > 70:
            print("\n🚨 灾难性结论: 严重的身份过拟合！")
            print("模型基本是个“人脸识别器”。它极大概率只是记住了数据集里抑郁症患者的脸型和长相，临床泛化能力堪忧。")
        elif overfit_rate > 30:
            print("\n⚠️ 警告结论: 中度身份纠缠。")
            print("模型部分依赖了静态长相特征，未能完全将“病理动态”解耦。建议在训练时加入时间维度的 Data Augmentation（如帧乱序、帧遮挡等）来逼迫模型学习动态。")
        else:
            print("\n✅ 优秀结论: 模型学到了真正的病理！")
            print("当动态消失时，模型不再认为患者抑郁，这证明它确实在捕捉诸如微表情迟缓、眼动异常等时序病理特征，而非单纯认脸。")
    else:
        print("模型未能正确识别任何原始样本，无法进行探针分析。")


if __name__ == '__main__':
    main()
