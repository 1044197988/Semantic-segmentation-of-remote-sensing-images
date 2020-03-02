import cv2
import random
import numpy as np
import os
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from FCN8S import dice_coef
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

basePath="C:\\Users\Administrator\Desktop\Project\\";
TEST_SET = ['1.png']

image_size = 32

classes=[0.0,1.0,2.0,3.0,4.0,15.0] 
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False,default="FCN.h5",
        help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
        help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    # load the trained convolutional neural network
    print("载入网络权重中……")
    model = load_model(args["model"],custom_objects={'dice_coef': dice_coef})
    stride = args['stride']
    print("进行预测分割拼图中……")
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread(basePath+'train\\' + path)
        h,w,_ = image.shape
        padding_h = (h//stride + 1) * stride 
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:3]
                ch,cw,_ = crop.shape
                #print(ch,cw,_)
                if ch != 32 or cw != 32:
                    print('尺寸不正确,请检查!')
                    continue
                crop = np.expand_dims(crop, axis=0) 
                pred = model.predict(crop,verbose=2)
                pred=np.argmax(pred,axis=3)
                pred=pred.flatten()
                pred = labelencoder.inverse_transform(pred)
                pred = pred.reshape((32,32)).astype(np.uint8)
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]

        cv2.imwrite(basePath+'predict/'+path,mask_whole[0:h,0:w])
        
    

    
if __name__ == '__main__':
    A=time.time()
    args = args_parse()
    predict(args)
    B=time.time()
    print("运行时长:%.1f" % float(B-A)+"s")


