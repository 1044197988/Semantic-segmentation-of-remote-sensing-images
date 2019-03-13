#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import argparse
import numpy as np  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation  
from tensorflow.keras.utils import to_categorical  
from tensorflow.keras.preprocessing.image import img_to_array  
from tensorflow.keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm

 #filepath ='C:\\Users\Administrator\Desktop\Project\src\\'

#import tensorflow.keras.backend.tensorflow_backend as KTF

#进行配置，使用80%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

# 设置session
tf.keras.backend.set_session(session)

#设置使用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7  
np.random.seed(seed)  
  
#设置图像大小
img_w = 32
img_h = 32

#分类
n_label=6
classes=[0.0,17.0,34.0,51.0,68.0,255.0]
labelencoder = LabelEncoder()  
labelencoder.fit(classes)
#训练批次和每次数据量
EPOCHS = 5
BS = 32
#图像最大值
divisor=255.0
#图像根路径
filepath ='C:\\Users\Administrator\Desktop\Project\src\\'

#读取图片
def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / divisor
    return img

#获取训练数据和测试数据地址
def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'train'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# 生成训练数据
def generateData(batch_size,data=[]):
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'train/' + url)
            img = img_to_array(img) 
            train_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))  
            # print label.shape  
            train_label.append(label)  
            if batch % batch_size==0: 
                #转为数组
                train_data = np.array(train_data)  
                train_label = np.array(train_label).flatten()      #拍平
                train_label = labelencoder.transform(train_label)  
                train_label = to_categorical(train_label, num_classes=n_label)  #编码输出便签
                train_label = train_label.reshape((batch_size,img_w * img_h,n_label))
                #print("1:"+str(train_label.shape))
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
                #print("2:"+str(len(train_label)))
                
# data for validation

#生成有效的数据
def generateValidData(batch_size,data=[]):
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'train/' + url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))  
            #print(label.shape)  
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)
                valid_label = valid_label.reshape((batch_size,img_w * img_h,n_label))
                #print("3:"+str(valid_label.shape))
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0
                #print("4:"+str(len(valid_label)))

#定义模型-网络模型
def SegNet():
    model = Sequential()  
    #encoder  
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_w,img_h,3),padding='same',activation='relu',data_format='channels_last'))
    model.add(BatchNormalization())  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2,2)))
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    #(8,8)  
    #decoder  
    model.add(UpSampling2D(size=(2,2)))  
    #(16,16)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(32,32)  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(64,64)  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(128,128)  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(UpSampling2D(size=(2, 2)))  
    #(256,256)  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h,3), padding='same', activation='relu',data_format='channels_last'))
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
    model.add(BatchNormalization())  
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))  
    model.add(Reshape((img_w*img_h,n_label)))
    #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)  
    #model.add(Permute((2,1)))  
    model.add(Activation('softmax'))  
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
    model.summary()  
    return model

#开始训练
def train(args):
    model = SegNet()
    
    modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',save_best_only=True,mode='max')
    callable = [modelcheck,tf.keras.callbacks.TensorBoard(log_dir='.')]  
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=(train_numb//BS),epochs=EPOCHS,verbose=2,
                    validation_data=generateValidData(BS,val_set),validation_steps=(valid_numb//BS),callbacks=callable,max_queue_size=1)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

#获取参数
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augment", help="using data augment or not",
                    action="store_true", default=False)
    ap.add_argument("-m", "--model", required=False,default="segnet.h5",
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args

#运行程序
if __name__=='__main__':  
    args = args_parse()
    train(args)
    print("完成")
    #predict()
