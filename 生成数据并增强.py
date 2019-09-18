#coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 64
img_h = 64 

basePath="C:\\Users\Administrator\Desktop\Project\\";

image_sets = ['1.png']

#高斯噪声
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

#椒盐噪声
def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    
#旋转处理
def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb

#模糊处理
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

#添加噪点信息
def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
#添加数据
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    #对原图像做模糊处理
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb

#创建数据
def creat_dataset(image_num = 2000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        imgPath=basePath+'train\\' + image_sets[i];
        src_img = cv2.imread(imgPath)  # 3 channels
        #print("\n图片"+imgPath)
        labelPath=basePath+'label\\' + image_sets[i]
        #print("图片"+labelPath)
        label_img = cv2.imread(labelPath,cv2.IMREAD_GRAYSCALE)  # 1 channel
        #print(label_img)
        X_height,X_width,_ = src_img.shape
        print("\n")
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            
            visualize = np.zeros((64,64)).astype(np.uint8)
            visualize = label_roi *50
            
            cv2.imwrite((basePath+'src//visualize//%d.png' % g_count),visualize)
            cv2.imwrite((basePath+'src//train//%d.png' % g_count),src_roi)
            cv2.imwrite((basePath+'src//label//%d.png' % g_count),label_roi)
            count += 1 
            g_count += 1
            if count%100==0:
                print("已经生成"+ str(count) +"张图片")

            
    

if __name__=='__main__':  
    creat_dataset(mode='augment')
