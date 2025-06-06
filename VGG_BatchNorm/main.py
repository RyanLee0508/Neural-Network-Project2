#%% 导入模块
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch.optim.lr_scheduler import StepLR
import os
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Dropout, VGG_A_BN  


# tensorboard
writer = SummaryWriter('./logs/')
# 训练设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 参数定义
EPOCH = 60
BATCH_SIZE = 64
LR = 0.0001
n_train_samples = 5000   # 控制训练集大小
n_test_samples = 1000    # 控制测试集大小

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
model = VGG_A_Light().to(device)
optim = torch.optim.Adam(model.parameters(), LR)
scheduler = StepLR(optim, step_size=30, gamma=0.1)  # 每 30 个 epoch 学习率 ×0.1
lossf = nn.CrossEntropyLoss()
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
            loss += lossf(output, targets)
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
        model_dir = 'best_models/VGG_A_Light/'

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
        loss = lossf(output, targets)
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
# 打印并保存最优模型的信息
model_saved_show = ' '.join(model_saved_list)
print('| BEST-MODEL | '+model_saved_show)
with open('model.txt', 'a') as f:
    f.write(model_saved_show+'\n')


