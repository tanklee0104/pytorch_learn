import torch
from torch import nn
from model import *
import torchvision
from torch.utils.data import DataLoader

# datasets
train_data = torchvision.datasets.CIFAR10("dataset_CIFAR10", train = True, 
                    transform = torchvision.transforms.ToTensor(), download= True)
test_data = torchvision.datasets.CIFAR10("dataset_CIFAR10", train = False, 
                    transform = torchvision.transforms.ToTensor(), download= True)

# length of training data and test data
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集长度为：{}".format(train_data_size))
print("测试集长度为：{}".format(test_data_size))

# Dataloader
train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

# neural network
tank = Tank() 

# loss function
loss_function = nn.CrossEntropyLoss()

# optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(tank.parameters(), lr = learning_rate)

# parameters for training network
total_train_steps = 0
total_test_steps = 0
# epoch 
epoch = 10

for i in range(epoch):
    print("------------第{}轮训练开始-------------".format(i + 1))

    # train
    for data in train_dataloader:
        imgs, targets = data
        outputs = tank(imgs)
        loss = loss_function(outputs, targets)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_steps += 1
        if total_train_steps % 100 == 0:
            print("训练次数：{},  loss={}".format(total_train_steps, loss.item()))

    # 测试步骤
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tank(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))

    torch.save(tank, "model_train/tank_{}.pth".format(i+1))
    # torch.save(tank.state_dict(), "model_test/tank_{}.pth".format(i + 1))
    print("模型已保存")
