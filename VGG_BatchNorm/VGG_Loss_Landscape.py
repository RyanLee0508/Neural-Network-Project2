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

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports_BN', 'figures_BN')
models_path = os.path.join(home_path, 'reports_BN', 'models_BN')

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
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad_list = []  # use this to record the gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

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
            # Record gradient norm for the first layer (example)
            grad_list.append(model.features[0].weight.grad.norm().item())
            
        if scheduler is not None:
            scheduler.step()

        losses_list.append(np.mean(loss_list))
        grads.append(np.mean(grad_list))
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] = np.mean(loss_list)
        train_accuracy_curve[epoch] = get_accuracy(model, train_loader)
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader)

        axes[0].plot(learning_curve, label='Training Loss')
        axes[0].legend()
        axes[1].plot(train_accuracy_curve, label='Training Accuracy')
        axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
        axes[1].legend()

        plt.savefig(os.path.join(figures_path, f'training_progress_BN_epoch_{epoch}.png'))
        plt.close()

        if val_accuracy_curve[epoch] > max_val_accuracy:
            max_val_accuracy = val_accuracy_curve[epoch]
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)

    return losses_list, grads

# Train your model
# feel free to modify
epo = 20
loss_save_path = os.path.join(models_path, 'losses_BN')
grad_save_path = os.path.join(models_path, 'grads_BN')
best_model_path = os.path.join(models_path, 'best_model_BN.pth')

os.makedirs(loss_save_path, exist_ok=True)
os.makedirs(grad_save_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)  

set_random_seeds(seed_value=2020, device=device)
model = VGG_A_BN()
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
losses_list, grads = train(model, optimizer, criterion, train_loader, val_loader, scheduler=scheduler, epochs_n=epo, best_model_path=best_model_path)

np.savetxt(os.path.join(loss_save_path, 'loss_BN.txt'), losses_list, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads_BN.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
for loss in losses_list:
    min_curve.append(np.min(loss))
    max_curve.append(np.max(loss))
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(losses, grads, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Landscape')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(grads, label='Gradient Norm', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Landscape')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_and_grad_landscape_BN.png'))
    plt.close()

plot_loss_landscape(losses_list, grads, figures_path)

