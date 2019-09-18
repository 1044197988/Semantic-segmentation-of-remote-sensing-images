# Semantic-segmentation-of-remote-sensing-image   <br>基于深度学习关于遥感影像的语义分割 
首先看一下数据集，包含原始影像与标签，实际的分辨率很大，这个只是缩略图。<br>

此代码库可在Tensorflow下keras环境运行，在Tensorflow1.12及Tensorflow2.0测试运行，代码更改后，更适合于Tensorflow2.0<br>

![train](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Data/train.png)
![label](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Data/label.png)
<br>

## Unet、FPN模型及嵌入相关模块后的结果：
通过fit_generator运行，所以生成器需要自己编写，FCN8S与Segnet均为序列式模型与Keras的Model类有些不同，可以调用更多的方法。<br>
展示一下Unet模型及FPN模型在此数据集上的结果，结果比Segnet与FCN好太多，所以就在这里不对比Segnet与FCN了。<br>

其中Unet未经过预训练，其他集成的模块都经过了Imagenet预训练。

### 准确率对比：
![ACC](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Image/Acc.png)

### Iou对比：
![IOU](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Image/IOU.png)

### Loss对比：
![Loss](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Image/Loss.png)

### 参数对比：
![参数](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images/blob/master/Image/dd.png)

## 代码运行：
弄好数据集后，需要切割，切割的话这个可以参考一下生成数据并增强.py,更改相关参数即可<br>
然后通过里面的Segnet的训练程序启动即可，需要修改参数。<br>
这个只是序列式模型的预测与训练，主要还是看模型。
