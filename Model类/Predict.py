import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from Unet import unet
import cv2

def predict(TEST_SET,image_size):
    print("载入网络权重中……")
    try:
        model = Unet(6,(32,32,4),0.001,0.00001) # build UNet
        model.load_weights('UnetDen169SGD.h5')
    except:
        print("载入失败！")
    stride = image_size
    print("进行预测分割拼图中……")
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        image = Image.open(basePath+path)
        w,h = image.size
        padding_h = (h//stride + 1) * stride
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,4),dtype=np.uint8)
        image=img_to_array(image)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:4]
                ch,cw,_ = crop.shape
                if ch != 32 or cw != 32:
                    print('尺寸不正确,请检查!')
                    continue

                crop = np.expand_dims(crop, axis=0) 
                pred = model.predict(crop)
                pred = np.argmax(pred,axis=3)
                pred = pred.flatten()
                pred = labelencoder.inverse_transform(pred)
                pred = pred.reshape((32,32)).astype(np.uint8)
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]

        cv2.imwrite(basePath2+'predict\\%s'%path,mask_whole[0:h,0:w])
        
if __name__ == '__main__':
    classes=[0.0, 22.0, 59.0, 73.0, 82.0, 92.0, 111.0, 117.0, 146.0, 161.0, 165.0, 177.0, 179.0, 180.0, 191.0, 192.0]
    labelencoder = LabelEncoder()  
    labelencoder.fit(classes)
    TEST_SET = ["1.png","2.png"]
    predict(TEST_SET,image_size)




