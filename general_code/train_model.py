from typing import Callable, Iterable, List, Union
from cmath import isnan

from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到Python路径
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from general_code import utils


def init_weights(m: nn.Module):
    types = [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]
    if type(m) in types:
        nn.init.xavier_uniform_(m.weight)


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate the number of correct predictions."""
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def try_gpu(i: int = 0) -> torch.device:
    """Return the best device available.

    :param i: ID of a specific GPU you would like to use.
    :returns: A GPU if it's available. Otherwise the CPU.
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def test(
    net: nn.Module,
    test_iter: Iterable,
    loss: Callable[[torch.Tensor, torch.Tensor], float],
    device: Union[torch.device, None] = None,
    task_name: str = "Untitled task",
    calc_accuracy: bool = False,
):
    """Test the model. Not task-specific."""
    if device is None:
        device = try_gpu()
    print(f"Start testing...")
    print(f"Task name: {task_name}, Device: {device}")

    net.to(device)
    net.eval()

    metric = [0, 0, 0]
    # metric:
    # 0. total loss;
    # 1. total num. of correct predictions;
    # 2. total num. of data

    with torch.no_grad():
        for X, y in tqdm(test_iter):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)

            metric[0] += l * X.shape[0]
            if calc_accuracy:
                metric[1] += accuracy(y_hat, y)
            metric[2] += X.shape[0]

    test_loss = metric[0] / metric[2]
    if calc_accuracy:
        test_acc = metric[1] / metric[2]
        print(f"test loss {test_loss:.3f}, test acc {test_acc:.3f}")
        return test_loss, test_acc
    else:
        print(f"test loss {metric[0] / metric[2]:.3f}")
        return test_loss


def train(
    net: nn.Module,
    train_iter: Iterable,
    num_epochs: int,
    lr: float,
    loss: Callable[[torch.Tensor, torch.Tensor], float],
    weight_decay: float,
    momentum: float,
    calc_accuracy: bool = False,
    device: Union[torch.device, None] = None,
    task_name: str = "Untitled task",
    lr_decay: bool = False,
    val_iter: Union[Iterable, None] = None,  # 新增：验证集
    patience: int = 10,  # 新增：早停耐心值
    min_delta: float = 0.001,  # 新增：最小改善值
    save_best: bool = True,  # 新增：是否保存最佳模型
) -> None:
    """Train the model. Not task-specific."""
    if device is None:
        device = try_gpu()
    print(f"Start training...")
    print(f"Task name: {task_name}, Device: {device}")
    print(f"Learning rate: {lr}, Weight decay: {weight_decay}, Momentum: {momentum}")

    # 新增：早停机制相关变量
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    net.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
    )
    if lr_decay:
        scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, mode="min", patience=3
        )
    train_l_ls, train_acc_ls = [], []
    # 新增：记录验证损失
    val_l_ls, val_acc_ls = [], []
    num_batch = len(train_iter)

    epoch_bar = tqdm(
        range(num_epochs),
        position=0,
        desc="epoch",
        leave=False,
        colour="green",
        ncols=80,
    )
    batch_bar = tqdm(
        range(num_batch),
        position=1,
        desc="batch",
        colour="red",
        total=num_batch,
        ncols=80,
    )

    for epoch in range(num_epochs):
        batch_bar.refresh()
        batch_bar.reset()

        metric = [0, 0, 0]
        # metric:
        # 0. total loss;
        # 1. total num. of correct predictions;
        # 2. total num. of data

        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            if loss.__class__.__name__ == "CrossEntropyLoss":
                y = y.long()  # 转换为Long类型
            y_hat = net(X)
            l = loss(y_hat, y)
            if torch.any(torch.isnan(y_hat)) or torch.any(torch.isnan(l)):
                raise Exception(f"NAN in epoch: {epoch}, iter: {i}, loss: {l.item()}")
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric[0] += l * X.shape[0]
                if calc_accuracy:
                    metric[1] += accuracy(y_hat, y)
                metric[2] += X.shape[0]
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            train_l_ls.append(train_l)
            train_acc_ls.append(train_acc)
            batch_bar.update()

        # 新增：验证阶段
        if val_iter is not None:
            net.eval()
            val_metric = [0, 0, 0]
            with torch.no_grad():
                for X_val, y_val in val_iter:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    if loss.__class__.__name__ == "CrossEntropyLoss":
                        y_val = y_val.long()
                    y_hat_val = net(X_val)
                    l_val = loss(y_hat_val, y_val)
                    val_metric[0] += l_val * X_val.shape[0]
                    if calc_accuracy:
                        val_metric[1] += accuracy(y_hat_val, y_val)
                    val_metric[2] += X_val.shape[0]

            val_loss = val_metric[0] / val_metric[2]
            val_l_ls.append(val_loss)

            if calc_accuracy:
                val_acc = val_metric[1] / val_metric[2]
                val_acc_ls.append(val_acc)
                tqdm.write(
                    f"epoch: {epoch}, train loss {train_l:.3f}, train acc {train_acc:.3f}, "
                    f"val loss {val_loss:.3f}, val acc {val_acc:.3f}"
                )
            else:
                tqdm.write(
                    f"epoch: {epoch}, train loss {train_l:.3f}, val loss {val_loss:.3f}"
                )

            # 早停机制判断
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型状态
                if save_best:
                    best_model_state = net.state_dict().copy()
                tqdm.write(f"Validation loss improved to {val_loss:.4f}")
            else:
                patience_counter += 1
                tqdm.write(f"No improvement for {patience_counter}/{patience} epochs")

            # 检查是否应该早停
            if patience_counter >= patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch}")
                # 恢复最佳模型
                if save_best and best_model_state is not None:
                    net.load_state_dict(best_model_state)
                    tqdm.write("Best model restored")
                break
        else:
            # 如果没有验证集，只打印训练信息
            if calc_accuracy:
                tqdm.write(
                    f"epoch: {epoch}, train loss {train_l:.3f}, train acc {train_acc:.3f}"
                )
            else:
                tqdm.write(f"epoch: {epoch}, train loss {train_l:.3f}")

        if lr_decay:
            scheduler_lr.step(train_l)

        epoch_bar.update()

    # 如果训练完成但从未触发早停，并且有保存最佳模型，则恢复最佳模型
    if save_best and best_model_state is not None and val_iter is not None:
        net.load_state_dict(best_model_state)

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(torch.tensor(train_l_ls).cpu().numpy(), label="Training Loss")
    if val_iter is not None:
        plt.plot(val_l_ls, label="Validation Loss")
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{task_name} Loss")
    plt.grid(True, alpha=0.3)

    if calc_accuracy:
        plt.subplot(1, 2, 2)
        plt.plot(torch.tensor(train_acc_ls).cpu().numpy(), label="Training Accuracy")
        if val_iter is not None and len(val_acc_ls) > 0:
            plt.plot(val_acc_ls, label="Validation Accuracy")
            plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{task_name} Accuracy")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"./{task_name}_training_curves.png")

    if calc_accuracy:
        if val_iter is not None:
            print(f"Best val loss {best_val_loss:.3f}, val acc {val_acc_ls[-1]:.3f}")
        print(f"Final train loss {train_l:.3f}, train acc {train_acc:.3f}")
        return train_l, train_acc
    else:
        if val_iter is not None:
            print(f"Best val loss {best_val_loss:.3f}")
        print(f"Final train loss {train_l:.3f}")
        return train_l


def get_full_img_pred(
    net: nn.Module,
    pil_image: Image.Image,
    num_sample: int = 3,
    side_length: int = 105,
    device: Union[torch.device, None] = None,
):
    """Randomly cup `num_sample` samples from `pil_image` and make prediction for each of the samples."""
    pil_image = pil_image.convert("L")
    h, w = pil_image.height, pil_image.width
    if h > w:
        pil_image = pil_image.resize((side_length, int(h / (w / side_length))))
    else:
        pil_image = pil_image.resize((int(w / (h / side_length)), side_length))

    samples = utils.image_sampling(pil_image, num_sample, side_length, side_length)

    if device is None:
        device = try_gpu()
    samples = utils.img_to_tensor(samples)
    samples = samples.to(device)
    net = net.to(device)
    pred = net(samples)

    return pred.cpu().tolist()
