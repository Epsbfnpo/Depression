import os
import torch
import yaml
import random
import numpy as np
from tqdm import tqdm
from models import DepMamba
from datasets import get_dvlog_dataloader, get_lmvd_dataloader


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_frankenstein_sample(feat_dep, feat_heal):
    """
    将抑郁症样本和健康样本沿时间维度截断对齐，并进行模态交叉拼接
    feat_dep: [T1, D] (抑郁症特征)
    feat_heal: [T2, D] (健康人特征)
    视觉维度: 0:136, 音频维度: 136:
    """
    # 1. 时间步对齐 (取最小值)
    min_t = min(feat_dep.shape[0], feat_heal.shape[0])
    f_dep = feat_dep[:min_t, :]
    f_heal = feat_heal[:min_t, :]

    # 2. 提取子模态
    v_dep, a_dep = f_dep[:, :136], f_dep[:, 136:]
    v_heal, a_heal = f_heal[:, :136], f_heal[:, 136:]

    # 3. 缝合弗兰肯斯坦样本
    # Frank 1: 抑郁画面 + 健康声音
    frank_vDep_aHeal = np.concatenate([v_dep, a_heal], axis=1)
    # Frank 2: 健康画面 + 抑郁声音
    frank_vHeal_aDep = np.concatenate([v_heal, a_dep], axis=1)

    # 截断后的原始对照组
    trunc_dep = np.concatenate([v_dep, a_dep], axis=1)
    trunc_heal = np.concatenate([v_heal, a_heal], axis=1)

    return trunc_dep, trunc_heal, frank_vDep_aHeal, frank_vHeal_aDep


def run_probe():
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 读取配置文件
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config['dataset']
    data_dir = os.path.join(config['data_dir'], dataset_name)
    save_dir = config['save_dir']

    # 2. 初始化模型
    print(f"[*] 初始化模型 DepMamba (Dataset: {dataset_name})...")
    if dataset_name == 'lmvd':
        net = DepMamba(**config['mmmamba_lmvd']).to(device)
    else:
        net = DepMamba(**config['mmmamba']).to(device)

    # 加载训练好的权重 (默认取 iteration 0 的 best_model)
    model_path = f"{save_dir}/{dataset_name}_DepMamba_0/checkpoints/best_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到权重文件：{model_path}。请先运行 main.py 训练模型！")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # 3. 加载测试集数据提取特征
    print("[*] 正在加载测试集样本用于缝合...")
    if dataset_name == 'dvlog':
        test_dataset = get_dvlog_dataloader(data_dir, "test", batch_size=1, gender="both").dataset
    else:
        test_dataset = get_lmvd_dataloader(data_dir, "test", batch_size=1, gender="both").dataset

    dep_samples = []
    heal_samples = []

    for i in range(len(test_dataset)):
        feat, label = test_dataset[i]
        if label == 1:
            dep_samples.append(feat)
        else:
            heal_samples.append(feat)

    print(f"    找到抑郁样本: {len(dep_samples)} 个, 健康样本: {len(heal_samples)} 个")

    # 4. 构建探针进行交叉测试
    num_pairs = min(len(dep_samples), len(heal_samples))
    # 打乱样本以获得随机配对
    random.shuffle(dep_samples)
    random.shuffle(heal_samples)

    results = {
        "Base_Dep": [],     # 原生抑郁
        "Base_Heal": [],    # 原生健康
        "Frank_vDep_aHeal": [],  # 画面抑郁 + 声音健康
        "Frank_vHeal_aDep": []   # 画面健康 + 声音抑郁
    }

    print(f"[*] 开始进行跨模态反事实缝合探针测试 (测试对数量: {num_pairs})...")
    with torch.no_grad():
        for i in tqdm(range(num_pairs)):
            f_dep, f_heal = dep_samples[i], heal_samples[i]

            t_dep, t_heal, f_vD_aH, f_vH_aD = create_frankenstein_sample(f_dep, f_heal)

            # 转 Tensor 并增加 Batch 维度，移至 GPU
            tensors = {
                "Base_Dep": torch.from_numpy(t_dep).unsqueeze(0).to(device),
                "Base_Heal": torch.from_numpy(t_heal).unsqueeze(0).to(device),
                "Frank_vDep_aHeal": torch.from_numpy(f_vD_aH).unsqueeze(0).to(device),
                "Frank_vHeal_aDep": torch.from_numpy(f_vH_aD).unsqueeze(0).to(device)
            }

            for key, tensor_input in tensors.items():
                # 构造 padding_mask (全 1，因为没有 padding)
                mask = torch.ones(1, tensor_input.shape[1]).to(device)

                # 前向传播
                logits = net(tensor_input, mask)
                prob = torch.sigmoid(logits).item()  # 转化为 0~1 的患病概率
                results[key].append(prob)

    # 5. 统计与诊断输出
    print("\n================= 探针诊断报告 =================\n")
    avg_probs = {k: np.mean(v) for k, v in results.items()}

    print(f"[对照组] 纯抑郁样本 (V_dep + A_dep) 平均预测概率: {avg_probs['Base_Dep'] * 100:.2f}%")
    print(f"[对照组] 纯健康样本 (V_heal + A_heal) 平均预测概率: {avg_probs['Base_Heal'] * 100:.2f}%\n")

    print(f"[实验组 1] 抑郁画面 + 健康声音 (V_dep + A_heal): {avg_probs['Frank_vDep_aHeal'] * 100:.2f}%")
    print(f"[实验组 2] 健康画面 + 抑郁声音 (V_heal + A_dep): {avg_probs['Frank_vHeal_aDep'] * 100:.2f}%\n")

    # 诊断逻辑分析
    shift_audio_driven = avg_probs['Frank_vHeal_aDep'] - avg_probs['Base_Heal']
    shift_visual_driven = avg_probs['Frank_vDep_aHeal'] - avg_probs['Base_Heal']

    print("【病理结论分析】：")
    if avg_probs['Frank_vHeal_aDep'] > 0.5 and avg_probs['Frank_vDep_aHeal'] < 0.5:
        print(">> 🚨 确诊捷径学习 (Shortcut Learning)！模型患有严重的【音频依赖症】。")
        print(">> 模型只听声音就能判断抑郁，视觉画面即使是抑郁症患者，也会被诊断为健康。视觉分支在陪跑！")
    elif avg_probs['Frank_vDep_aHeal'] > 0.5 and avg_probs['Frank_vHeal_aDep'] < 0.5:
        print(">> 🚨 确诊捷径学习 (Shortcut Learning)！模型患有严重的【视觉依赖症】。")
        print(">> 声音分支形同虚设。")
    else:
        print(">> ✅ 未发现极端的单模态捷径病理。")
        print(">> 模型在面对冲突的模态时，预测概率出现了拉扯（没有倒向任何一个极端），说明它确实在进行多模态信息的综合 (Synergy)。")
    print("==================================================")


if __name__ == "__main__":
    run_probe()
