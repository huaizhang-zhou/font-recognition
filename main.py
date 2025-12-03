import json

import torch

from general_code import utils
from general_code import image_generation
from font_recognition import SCAE
from font_recognition import CNN


def main():
    utils.init_seed(42)

    fonts_path = "./dataset/fonts"
    generated_image_path = "./dataset/generated_images"
    generated_label_path = "./dataset/generated_images/labels.csv"
    real_image_path = "./dataset/real_images"
    models_path = "./dataset/models"

    total_num = 10000
    language = "en"
    gen_batch_size = 5
    sample_batch_size = 2
    sample_num = 2
    sample_width = 105
    sample_height = 105

    train_batch_size = 50
    num_epochs = 200

    generate_image = True

    fonts_ls = image_generation.get_fonts_list(fonts_path)
    num_fonts = len(fonts_ls)
    SCAE_model_path = f"{models_path}/SCAE_{language}_{num_fonts}.pth"
    CNN_model_path = f"{models_path}/CNN_{language}_{num_fonts}.pth"
    font_dict_path = f"{models_path}/font_dict_{language}_{num_fonts}.json"

    if generate_image:
        image_generation.generate_images(
            total_num=total_num,
            language=language,
            fonts_path=fonts_path,
            gen_batch_size=gen_batch_size,
            gen_image_path=generated_image_path,
            label_df_path=generated_label_path,
            need_save=True,
            need_return=False,
        )

        image_generation.saved_images_sampling(
            total_num=total_num,
            img_path=generated_image_path,
            sample_path=generated_image_path,
            sample_batch_size=sample_batch_size,
            sample_num=sample_num,
            width=sample_width,
            height=sample_height,
            need_save=True,
            need_return=False,
        )

    # 修改：获取训练集和验证集
    SCAE_train_iter, SCAE_val_iter, _ = SCAE.get_SCAE_train_val_dataloader(
        batch_size=train_batch_size,
        total_num=total_num,
        sample_num=sample_num,
        generated_img_path=generated_image_path,
        generated_label_path=generated_label_path,
        real_img_path=real_image_path,
        val_split=0.2,
    )
    SCAE_net = SCAE.SCAE()
    SCAE.train_SCAE(
        SCAE_net,
        SCAE_train_iter,
        num_epochs,
        val_iter=SCAE_val_iter,  # 传入验证集
        patience=25,  # 设置耐心值
        min_delta=0.001,  # 最小改善值
    )
    torch.save(SCAE_net, SCAE_model_path)

    # 修改：获取训练集和验证集
    CNN_train_iter, CNN_val_iter, CNN_full_dataset = CNN.get_CNN_train_val_dataloader(
        batch_size=train_batch_size,
        total_num=total_num,
        sample_num=sample_num,
        generated_img_path=generated_image_path,
        generated_label_path=generated_label_path,
        val_split=0.2,
    )
    CNN_net = CNN.CNN(SCAE_net.encoder, num_fonts)
    CNN.train_CNN(
        CNN_net,
        CNN_train_iter,
        num_epochs,
        val_iter=CNN_val_iter,  # 传入验证集
        patience=25,  # 设置耐心值
        min_delta=0.001,  # 最小改善值
    )
    torch.save(CNN_net, CNN_model_path)

    font_dict = CNN_full_dataset.font_dict

    with open(font_dict_path, "w") as f:
        json.dump(font_dict, f)


if __name__ == "__main__":
    main()
