#%% 导入模块
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch.optim.lr_scheduler import StepLR
import os
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Dropout, VGG_A_BN,VGG_A_LeakyReLU
import matplotlib.pyplot as plt

# 可视化卷积核
def visualize_filters(model, layer_name="features.0.weight"):
    # 获取第一层卷积层的权重
    conv_layer = model.features[0]  # 假设第一个是 Conv2d 层
    weights = conv_layer.weight.data.cpu()

    # 可视化前 64 个 filter
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            ax.imshow(weights[i, 0].numpy(), cmap='gray')
            ax.axis('off')
    plt.suptitle(f"Filters of {layer_name}")
    plt.show()

# 损失函数曲面（Loss Landscape）可视化
def plot_loss_landscape(model, data_loader, lossf, direction1, direction2, steps=20):
    model.eval()
    loss_values = []

    with torch.no_grad():
        inputs, targets = next(iter(data_loader))
        inputs, targets = inputs.to(device), targets.to(device)

        # 获取原始参数（仅用于拷贝结构）
        original_params = [p.clone() for p in model.parameters()]

        for alpha in np.linspace(-1, 1, steps):
            row = []
            for beta in np.linspace(-1, 1, steps):

                # 保存当前参数副本，用于后续恢复
                backup_params = [p.clone() for p in model.parameters()]

                # 扰动参数
                for p, d1, d2 in zip(model.parameters(), direction1, direction2):
                    p.add_(alpha * d1 + beta * d2)

                # 前向传播 & 计算 loss
                output = model(inputs)
                targets_one_hot = torch.zeros_like(output).scatter_(1, targets.unsqueeze(1), 1)
                loss = lossf(output, targets_one_hot).item()
                row.append(loss)

                # 恢复原始参数
                for p, orig in zip(model.parameters(), original_params):
                    p.copy_(orig)

            loss_values.append(row)

    # 绘图部分不变
    X, Y = np.meshgrid(np.linspace(-1, 1, steps), np.linspace(-1, 1, steps))
    Z = np.array(loss_values)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.title("Loss Landscape")
    plt.xlabel("Direction 1")
    plt.ylabel("Direction 2")
    plt.show() 
    
# tensorboard
writer = SummaryWriter('./logs/')
# 训练设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 参数定义
EPOCH = 20
BATCH_SIZE = 64
LR = 0.0001
n_train_samples = 50000   # 控制训练集大小
n_test_samples = 10000    # 控制测试集大小

# 下载数据集
train_file = datasets.CIFAR10(
    root='./dataset/',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    download=True
)
test_file = datasets.CIFAR10(
    root='./dataset/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)


# 设置训练样本数量
train_subset = torch.utils.data.Subset(train_file, indices=range(n_train_samples))

# 制作数据加载器
train_loader = DataLoader(
    dataset=train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_subset = torch.utils.data.Subset(test_file, indices=range(n_test_samples))

test_loader = DataLoader(
    dataset=test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 创建模型
model = VGG_A_LeakyReLU().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = StepLR(optim, step_size=30, gamma=0.1)  # 每 30 个 epoch 学习率 ×0.1
lossf = nn.MSELoss()
# 定义计算整个训练集或测试集loss及acc的函数
def calc(data_loader):
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            targets_one_hot = torch.zeros_like(output).scatter_(1, targets.unsqueeze(1), 1)
            loss += lossf(output, targets_one_hot)
            correct += (output.argmax(1) == targets).sum()
            total += data.size(0)
    loss = loss.item()/len(data_loader)
    acc = correct.item()/total
    return loss, acc
# 训练过程打印函数
def show():
    # 定义全局变量
    if epoch == 0:
        global model_saved_list
        global temp
        temp = 0
    # 打印训练的EPOCH和STEP信息
    header_list = [
        f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
        f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}'
    ]
    header_show = ' '.join(header_list)
    print(header_show, end=' ')
    # 打印训练的LOSS和ACC信息
    loss, acc = calc(train_loader)
    writer.add_scalar('loss', loss, epoch+1)
    writer.add_scalar('acc', acc, epoch+1)
    train_list = [
        f'LOSS: {loss:.4f}',
        f'ACC: {acc:.4f}'
    ]
    train_show = ' '.join(train_list)
    print(train_show, end=' ')
    # 打印测试的LOSS和ACC信息
    val_loss, val_acc = calc(test_loader)
    writer.add_scalar('val_loss', val_loss, epoch+1)
    writer.add_scalar('val_acc', val_acc, epoch+1)
    test_list = [
        f'VAL-LOSS: {val_loss:.4f}',
        f'VAL-ACC: {val_acc:.4f}'
    ]
    test_show = ' '.join(test_list)
    print(test_show, end=' ')
    # 保存最佳模型
    if val_acc > temp:
        model_saved_list = header_list + train_list + test_list
        model_dir = 'best_models/AdamW/'

        # 如果目录不存在，则创建
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 保存模型
        with open(os.path.join(model_dir, 'model.pickle'), 'wb') as f:
            pickle.dump(model, f)

        temp = val_acc
# 训练模型
for epoch in range(EPOCH):
    start_time = time.time()
    for step, (data, targets) in enumerate(train_loader):
        optim.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        targets_one_hot = torch.zeros_like(output).scatter_(1, targets.unsqueeze(1), 1)
        loss = lossf(output, targets_one_hot)
        acc = (output.argmax(1) == targets).sum().item()/BATCH_SIZE
        loss.backward()
        optim.step()
        print(
            f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
            f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
            f'LOSS: {loss.item():.4f}',
            f'ACC: {acc:.4f}',
            end='\r'
        )
    show()
    scheduler.step()
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time-start_time)}')
    
# visualize_filters(model)  #  卷积核（Filter）可视化
# 生成随机方向
direction1 = [torch.randn_like(p) for p in model.parameters()]
direction2 = [torch.randn_like(p) for p in model.parameters()]

# 调用函数
plot_loss_landscape(model, train_loader, lossf, direction1, direction2, steps=20)

# 打印并保存最优模型的信息
model_saved_show = ' '.join(model_saved_list)
print('| BEST-MODEL | '+model_saved_show)
with open('model.txt', 'a') as f:
    f.write(model_saved_show+'\n')


