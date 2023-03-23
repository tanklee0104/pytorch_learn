from tracemalloc import start
import torch
import time
from torch import nn
# from model import *
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

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
class Tank(nn.Module):
    def __init__(self):
        super(Tank, self).__init__()
        self.model = Sequential(
            Conv2d(3,32,5,padding = 2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding = 2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding = 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
tank = Tank() 
if torch.cuda.is_available():
    tank = tank.cuda() # gpu正规写法


# loss function
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.cuda()

# optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(tank.parameters(), lr = learning_rate)

# parameters for training network
total_train_steps = 0
total_test_steps = 0
# epoch 
epoch = 20

for i in range(epoch):
    print("------------第{}轮训练开始-------------".format(i + 1))
    # train
    start_time = time.time()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tank(imgs)
        loss = loss_function(outputs, targets)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_steps += 1
        if total_train_steps % 100 == 0:
            print("训练次数：{},  loss={}".format(total_train_steps, loss.item()))
    end_time = time.time()
    print("time = {}s".format(end_time - start_time))

    # 测试步骤
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
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
