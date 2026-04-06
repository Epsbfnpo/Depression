from pathlib import Path
from typing import Union
import random

import numpy as np
import torch
from torch.utils import data


class DVlog(data.Dataset):
    def __init__(self, root: Union[str, Path], fold: str = "train", gender: str = "both", transform=None, target_transform=None, aug: bool = False):
        self.root = root if isinstance(root, Path) else Path(root)
        self.fold = fold
        self.gender = gender
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug

        self.v_features = []
        self.a_features = []
        self.labels = []

        with open(self.root / "labels.csv", "r") as f:
            for line in f:
                sample = line.strip().split(",")
                if not self.is_sample(sample):
                    continue

                s_id = sample[0]
                s_label = int(sample[1] == "depression")
                v_feature_path = self.root / s_id / f"{s_id}_visual.npy"
                a_feature_path = self.root / s_id / f"{s_id}_acoustic.npy"

                self.v_features.append(v_feature_path)
                self.a_features.append(a_feature_path)
                self.labels.append(s_label)

                if self.aug and self.fold == "train":
                    # 仅复制样本索引，不做对齐/拼接截断
                    for _ in range(5):
                        if random.random() > 0.5:
                            self.v_features.append(v_feature_path)
                            self.a_features.append(a_feature_path)
                            self.labels.append(s_label)

        print(f"ALL:{len(self.labels)}, Positive:{np.sum(self.labels)}, Negative:{len(self.labels) - np.sum(self.labels)}")

    def is_sample(self, sample) -> bool:
        gender, fold = sample[3], sample[4]
        if self.gender == "both":
            return fold == self.fold
        return (fold == self.fold) and (gender == self.gender)

    def __getitem__(self, i: int):
        v_feature = np.load(self.v_features[i])
        a_feature = np.load(self.a_features[i])
        label = self.labels[i]

        v_tensor = torch.tensor(v_feature, dtype=torch.float32)
        a_tensor = torch.tensor(a_feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            v_tensor = self.transform(v_tensor)
            a_tensor = self.transform(a_tensor)
        if self.target_transform is not None:
            label_tensor = self.target_transform(label_tensor)

        return {"video": v_tensor, "audio": a_tensor, "label": label_tensor}

    def __len__(self):
        return len(self.labels)


def get_dvlog_dataloader(root: Union[str, Path], fold: str = "train", batch_size: int = 1, gender: str = "both", transform=None, target_transform=None, aug: bool = True):
    dataset = DVlog(root, fold, gender, transform, target_transform, aug)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=(fold == "train"), drop_last=False)
    return dataloader
