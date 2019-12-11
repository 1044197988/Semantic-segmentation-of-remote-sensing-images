# Semantic-segmentation-of-remote-sensing-image   <br>基于深度学习关于遥感影像的语义分割 
首先看一下数据集，包含原始影像与标签，实际的分辨率很大，这个只是缩略图。<br>
影像数据是Landsat8卫星的，用五四三波段进行合成，并利用GS方法进行全色第八波段的融合。(Envi软件处理)<br>
标签是通过矢量图层以ArcGIS软件来处理生成的。<br>

此代码库可在Tensorflow下keras环境运行，在Tensorflow1.12及Tensorflow2.0测试运行，代码更改后，更适合于Tensorflow2.0<br>

![train](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Data/train.png)
![label](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Data/label.png)
<br>

## Unet、FPN模型及嵌入相关模块后的结果：
通过fit_generator运行，所以生成器需要自己编写，FCN8S与Segnet均为序列式模型与Keras的Model类有些不同，可以调用更多的方法。<br>
展示一下Unet模型及FPN模型在此数据集上的结果，结果比Segnet与FCN好太多，所以就在这里不对比Segnet与FCN了。<br>

其中Unet未经过预训练，其他集成的模块都经过了Imagenet预训练，并且测试都是通过划分数据来进行测试的，train75%，val25%。

### 准确率对比：
![ACC](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Image/Acc.png)

### Iou对比：
![IOU](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Image/IOU.png)

### Loss对比：
![Loss](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Image/Loss.png)

### 参数对比：
在相同的硬件条件下运行：
![参数](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Image/dd.png)

## 不同全色波段融合方法的结果：
在目录里可以看到，这个影响不大。

## Unet++模型结果：
### 混淆矩阵百分比：
![混淆矩阵](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Unet%2B%2B结果/confusion_matrix.png)
### 混淆矩阵像素数：
![混淆矩阵](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Unet%2B%2B结果/confusion_matrix_None.png)
### 相关统计：
当中1，2，3，4，5，15分别代表预测后的像素值，每个像素值代表一类
![1](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Unet%2B%2B结果/分类报告.png)
![2](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Unet%2B%2B结果/整体统计.png)
![3](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Unet%2B%2B结果/类别统计.png)

## 结果图的拼接痕迹问题：
我认为这个很大概率上跟模型的拟合能力、泛化能力有关，所以这个问题不用考虑，只要能够训练到所有的像素就可以了。
Unet等相关模型预测的图，“拼接痕迹”很小。

## 代码运行：
弄好数据集后，需要切割，切割的话这个可以参考一下生成数据并增强.py,更改相关参数即可。<br>
然后通过里面的Segnet的训练程序启动即可，需要修改参数。<br>
这个Segnet、FCN8S是用序列式类来实现的模型，所以预测的话是跟Model类有一点不相同，就是可以调用predict_classes的方法。<br>
Model类文件里提供了使用Model类的模型的预测方法。<br>
以上所有代码只是提供参考,训练其他数据的话很多参数需要自己修改，甚至生成器也需要更改，如果想对数据类别进行加权或使用样本权重，主要记得在生成器中做修改就可以了。

## CRF后处理:
模型预测完毕后，可以使用CRF进行后处理，CRF.py文件提供了相关代码参考，但未必保证结果可靠。

### GDAL
如果自己做的图包含多个波段(往往大于4个)，Opencv或PIL就不太顶用了，这时候GDAL就派上用场了<br>
例如我有一个十波段图像，用此函数读取后为numpy数组类,shape为[h,w,10]
```python
from osgeo import gdal
import numpy as np

def load_img(path):
    dataset = gdal.Open(path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    im_data = im_data.transpose((1,2,0)) #此步保证矩阵为channel_last模式
    return im_data
```

# 提示
我的该贡献库中，提供了一些分割的模型及相关指标与损失的代码<br>
[1044197988-TF.Keras-Commonly-used-models](https://github.com/1044197988/TF.Keras-Commonly-used-models)
