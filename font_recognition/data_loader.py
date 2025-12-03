from typing import Union

import torch
from torch.utils.data import dataloader, TensorDataset, dataset

# 添加项目根目录到Python路径
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from general_code.data_loader import TextAttributesRecognitionDataset
from general_code.utils import load_image, img_to_tensor


class FontRecognitionDataset(TextAttributesRecognitionDataset):
    def __init__(
        self,
        model_type: str,
        generated_img_path: str,
        generated_label_path: str,
        real_img_path: Union[str, None],
        total_num: int,
        sample_num: int,
    ):
        super(FontRecognitionDataset, self).__init__(
            model_type,
            generated_img_path,
            generated_label_path,
            real_img_path,
            total_num,
            sample_num,
        )
        if model_type == "SCAE":
            return  # SCAE
        elif model_type == "CNN":
            font_series = self.label_df["font"]
            font_ls = font_series.unique().tolist()
            self.font_dict = {font_ls[i]: i for i in range(len(font_ls))}
            self.labels = torch.tensor(
                font_series.apply(lambda x: self.font_dict[x]).tolist()
            )
        else:
            raise ValueError("Model type can only be `SCAE` or `CNN`.")

    def load_and_process_img(self, img_path: str):
        img = load_image(img_path)
        img = img.convert("L")
        img = img_to_tensor(img)

        return img


def get_dataloader_dataset(
    model_type: str,
    generated_img_path: str,
    generated_label_path: str,
    real_img_path: Union[str, None],
    total_num: int,
    sample_num: int,
    batch_size: int,
    shuffle: bool = True,
):
    train_dataset = FontRecognitionDataset(
        model_type,
        generated_img_path,
        generated_label_path,
        real_img_path,
        total_num,
        sample_num,
    )
    train_loader = dataloader.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    return train_loader, train_dataset


def split_dataset(
    dataset: dataset.Dataset, train_ratio: float = 0.8, random_seed: int = 42
):
    """将数据集拆分为训练集和验证集"""
    from torch.utils.data import random_split

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # 设置随机种子以确保可重复性
    torch.manual_seed(random_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


def get_train_val_dataloader(
    model_type: str,
    generated_img_path: str,
    generated_label_path: str,
    real_img_path: Union[str, None],
    total_num: int,
    sample_num: int,
    batch_size: int,
    val_split: float = 0.2,
    shuffle: bool = True,
    random_seed: int = 42,
):
    """获取训练集和验证集的数据加载器"""
    full_dataset = FontRecognitionDataset(
        model_type,
        generated_img_path,
        generated_label_path,
        real_img_path,
        total_num,
        sample_num,
    )

    train_dataset, val_dataset = split_dataset(
        full_dataset, train_ratio=1.0 - val_split, random_seed=random_seed
    )

    train_loader = dataloader.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    val_loader = dataloader.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False  # 验证集不需要shuffle
    )

    return train_loader, val_loader, full_dataset
