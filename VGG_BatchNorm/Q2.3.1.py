import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BN # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128
# 定义学习率列表
learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]  # [1e-3, 2e-3, 1e-4, 5e-4]
# add our package dir to path 
# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取当前脚本所在目录（即 Q2.3.1.py 所在目录）
script_dir = os.path.dirname(current_script_path)

# 构建 reports_Ir 目录路径
home_path = script_dir  # 改为直接使用脚本所在目录
figures_path = os.path.join(home_path, 'reports_Ir', 'figures_Ir')
# 模型保存路径
models_path = os.path.join(home_path, 'reports_Ir', 'models_Ir')
os.makedirs(models_path, exist_ok=True)

# 分别定义 VGG-A 和 VGG-A_BN 的最佳模型保存路径
best_model_paths_vgg_a = {
    lr: os.path.join(models_path, f'best_model_vgg_a_lr_{lr:.5f}.pth') for lr in learning_rates
}

best_model_paths_vgg_a_bn = {
    lr: os.path.join(models_path, f'best_model_vgg_a_bn_lr_{lr:.5f}.pth') for lr in learning_rates
}

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if torch.cuda.is_available():
    device_id = 0  # 使用第一块 GPU
    device = torch.device(f"cuda:{device_id}")
else:
    device = torch.device("cpu")
print(device)
print(torch.cuda.get_device_name(0))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_paths=None, learning_rates=[0.001]):
    model.to(device)
    learning_curves = {lr: [np.nan] * epochs_n for lr in learning_rates}
    train_accuracy_curves = {lr: [np.nan] * epochs_n for lr in learning_rates}
    val_accuracy_curves = {lr: [np.nan] * epochs_n for lr in learning_rates}
    max_val_accuracies = {lr: 0.0 for lr in learning_rates}  # 每个学习率单独记录最高准确率
    max_val_accuracy_epochs = {lr: 0 for lr in learning_rates}

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        optimizer.param_groups[0]['lr'] = lr
        best_acc_for_lr = 0.0
        best_epoch_for_lr = 0

        for epoch in tqdm(range(epochs_n), unit='epoch'):
            model.train()
            loss_list = []
            for data in train_loader:
                x, y = data
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                prediction = model(x)
                loss = criterion(prediction, y)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            avg_loss = np.mean(loss_list)
            train_acc = get_accuracy(model, train_loader)
            val_acc = get_accuracy(model, val_loader)

            learning_curves[lr][epoch] = avg_loss
            train_accuracy_curves[lr][epoch] = train_acc
            val_accuracy_curves[lr][epoch] = val_acc

            if val_acc > max_val_accuracies[lr]:
                max_val_accuracies[lr] = val_acc
                max_val_accuracy_epochs[lr] = epoch
                if best_model_paths and lr in best_model_paths:
                    torch.save(model.state_dict(), best_model_paths[lr])

    return learning_curves, train_accuracy_curves, val_accuracy_curves

# 打印每个学习率的训练结果
def print_learning_rate_results(loss_curves, train_acc_curves, val_acc_curves, label):
    print(f"\n=== Results for {label} ===")
    for lr in loss_curves:
        final_loss = loss_curves[lr][-1]
        final_train_acc = train_acc_curves[lr][-1]
        best_val_acc = max(val_acc_curves[lr])
        best_epoch = np.argmax(val_acc_curves[lr])
        print(f"Learning Rate: {lr:.5f}")
        print(f"  Final Loss       : {final_loss:.4f}")
        print(f"  Final Train Acc  : {final_train_acc * 100:.2f}%")
        print(f"  Best Val Acc     : {best_val_acc * 100:.2f}% at Epoch {best_epoch}")

# Train your model
# feel free to modify
epo = 10
# loss_save_path = os.path.join(models_path, 'losses_BN')
# grad_save_path = os.path.join(models_path, 'grads_BN')
# best_model_path = os.path.join(models_path, 'best_model_BN.pth')

# os.makedirs(loss_save_path, exist_ok=True)
# os.makedirs(grad_save_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)  

set_random_seeds(seed_value=2020, device=device)


# 训练VGG-A模型（无BN）
model_vgg_a = VGG_A()
optimizer_vgg_a = torch.optim.Adam(model_vgg_a.parameters(), lr=learning_rates[0])
criterion = nn.CrossEntropyLoss()
losses_vgg_a, train_acc_vgg_a, val_acc_vgg_a = train(
    model_vgg_a, optimizer_vgg_a, criterion, train_loader, val_loader,
    epochs_n=epo, best_model_paths=best_model_paths_vgg_a, learning_rates=learning_rates)

# 训练VGG-A_BN模型（有BN）
model_vgg_a_bn = VGG_A_BN()
optimizer_vgg_a_bn = torch.optim.Adam(model_vgg_a_bn.parameters(), lr=learning_rates[0])
losses_vgg_a_bn, train_acc_vgg_a_bn, val_acc_vgg_a_bn = train(
    model_vgg_a_bn, optimizer_vgg_a_bn, criterion, train_loader, val_loader,
    epochs_n=epo, best_model_paths=best_model_paths_vgg_a_bn, learning_rates=learning_rates)

# 输出每个学习率的训练结果
print_learning_rate_results(losses_vgg_a, train_acc_vgg_a, val_acc_vgg_a, "VGG-A")
print_learning_rate_results(losses_vgg_a_bn, train_acc_vgg_a_bn, val_acc_vgg_a_bn, "VGG-A_BN")
# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
# 计算最大和最小损失曲线
def calculate_max_min_curves(losses):
    max_curve = []
    min_curve = []
    for epoch in range(len(losses[learning_rates[0]])):
        max_loss = max([losses[lr][epoch] for lr in learning_rates])
        min_loss = min([losses[lr][epoch] for lr in learning_rates])
        max_curve.append(max_loss)
        min_curve.append(min_loss)
    return max_curve, min_curve

max_curve_vgg_a, min_curve_vgg_a = calculate_max_min_curves(losses_vgg_a)
max_curve_vgg_a_bn, min_curve_vgg_a_bn = calculate_max_min_curves(losses_vgg_a_bn)


# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(max_curve, min_curve, label, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(max_curve, label=f'{label} Max Loss')
    plt.plot(min_curve, label=f'{label} Min Loss')
    plt.fill_between(range(len(max_curve)), min_curve, max_curve, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Landscape for {label}')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'loss_landscape_{label}.png'))
    plt.close()

# 绘制VGG-A模型的损失景观
plot_loss_landscape(max_curve_vgg_a, min_curve_vgg_a, 'VGG-A', figures_path)

# 绘制VGG-A_BN模型的损失景观
plot_loss_landscape(max_curve_vgg_a_bn, min_curve_vgg_a_bn, 'VGG-A_BN', figures_path)

