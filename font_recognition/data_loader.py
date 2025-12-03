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
