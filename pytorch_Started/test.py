from PIL import Image
import torchvision
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lst_class = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']


img_path = "./dataset/airplane1.jpg"
image = Image.open(img_path)
# print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
image = transform(image)
# print(image.shape)


# Neural Network
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


model = torch.load("./model_train/tank_20.pth", map_location= torch.device("cpu"))  # 可以修改自己训练的路径
# print(model)
image = torch.reshape(image, (1,3,32,32))

model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))

class_pred = output.argmax(1).numpy()
# print(class_pred[0])
print(lst_class[class_pred[0]])