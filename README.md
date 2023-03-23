## B站小土堆的pytorch视频讲解对应的练习代码
* 视频讲解链接：https://www.bilibili.com/video/BV1hE411t7RN?p=1&vd_source=749692573c390d7b0d79d21140edc4aa


* 前期代码主要在study.ipynb文件中，包含了一些pytorch的基础操作
* 包括读取数据，transform，CIFAR10数据集，以及神经网络模型的讲解等等
* 神经网络模型又包括网路的架构，卷积、池化、Loss等等


* 后面的文件是一个完整的训练与测试项目,未添加softmax等激活函数，测试功能并不完善，仅用于学习测试
* 训练方式包括cpu与gpu，先运行train_xxx.py,再运行test.py测试


## 文件结构
```
|—— dataset: 包含一些测试图片， 
        注意：如果前期study.ipynb文件中有路径报错，原因是后面修改了文件夹名字，即 /dataset/birds --> /dataser/birds_image
|—— dataset_CIFAR10：CIFAR10数据集，包括训练集与测试集
|—— logs：TensorBoard的日志文件
|—— logs_model：同上
|—— model_train：训练模型保存的参数，效果最好的最后一次训练的文件
|—— model.py：网络模型,采用的结构是CIFAR10的网络架构，可以自行更改
|—— study.ipynb：前期代码，pytorch基础操作
|—— train_gpu_1.py：训练文件，训练设备为gpu
|—— test.py：测试文件
```

* 环境：pytorch > = 1.7.0
* pillow < = 8.2.0  使用8.2.0即可
* numpy
* cv2


