import cv2
import random
import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#进行配置，使用60%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
# 设置session
tf.keras.backend.set_session(session )

basePath="C:\\Users\Administrator\Desktop\Project\\";
TEST_SET = ['1.png']

image_size = 32

#分类
n_label=6
#classes=[0.0,1.0,2.0,3.0,4.0,5.0,6.0]
classes=[0.0,1.0,2.0,3.0,4.0,15.0]
#图像最大值
divisor=255.0

labelencoder = LabelEncoder()  
labelencoder.fit(classes)

#print(divisor)

#参数获取
def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False,default="segnet.h5",
        help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
        help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    # load the trained convolutional neural network
    #name_list = []
    print("载入网络权重中……")
    model = load_model(args["model"])
    stride = args['stride']
    print("进行预测分割拼图中……")
    #print(stride)
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #print(path)
        #load the image
        image = cv2.imread(basePath+'train\\' + path)
        h,w,_ = image.shape
        #print(h,w,_)
        padding_h = (h//stride + 1) * stride 
        padding_w = (w//stride + 1) * stride
        #print(padding_h,padding_w)
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / divisor
        padding_img = img_to_array(padding_img)
        #print('src:')
        #print(padding_img.shape)

        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:3]
                ch,cw,_= crop.shape
                #print(crop.shape)
                #print("1",ch,cw,_)
                
                if ch != 32 or cw != 32:
                    print('尺寸不正确,请检查!')
                    #print("2",ch,cw,_)
                    continue
                    
                crop = np.expand_dims(crop, axis=0)
                #print('crop:',crop.shape)
                pred = model.predict_classes(crop,verbose=2)
                #print(pred)
                #print(pred)
                #print(pred.shape)
                #print(pred[0])
                #print(pred[0].shape)
                pred = labelencoder.inverse_transform(pred[0])
                #for p in pred:
                    #name_list.append(name_dic[str(p)])
                #print(pred)
                #print(pred.shape)
                #print(np.unique(pred))
                #print(pred)
                pred = pred.reshape((32,32)).astype(np.uint8)
                #print('pred:',pred.shape)
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]    
        cv2.imwrite(basePath+'predict/'+path,mask_whole[0:h,0:w])
        #finally_result = pd.DataFrame({'file':test_name_list,'species':name_list})
        #print(finally_result)
        #finally_result.to_csv("result.csv",index=False)
if __name__ == '__main__':
    A=time.time()
    args = args_parse()
    predict(args)
    B=time.time()
    print("运行时长:%.1f" % float(B-A)+"s")

