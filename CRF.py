import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from PIL import Image
import numpy as np
#from tensorflow.keras.preprocessing.image import array_to_img
import cv2
"""
Function which returns the labelled image after applying CRF

"""

#Original_image = Image which has to labelled
#Annotated image = Which has been labelled by some technique( FCN in this case)
#Output_image = The final output image after applying CRF
#Use_2d = boolean variable 
#if use_2d = True specialised 2D fucntions will be applied
#else Generic functions will be applied

def crf(original_image, annotated_image,output_image, use_2d = True):
        
    ##将注释RGB颜色转换为单个32位整数
    #annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)
    #print(annotated_label.shape)
    
    # 将32位整数颜色转换为0、1、2、…标签。
    colors, labels = np.unique(annotated_label, return_inverse=True)
    print(len(labels))
    print(labels)
    print(len(colors))
    print(colors)
    n_labels = len(set(labels.flat))
    
    #创建一个32位颜色的映射
##    colorize = np.empty((len(colors), 3), np.uint8)
##    colorize[:,0] = (colors & 0x0000FF)
##    colorize[:,1] = (colors & 0x00FF00) >> 8
##    colorize[:,2] = (colors & 0xFF0000) >> 16
##    
##    #在带注释的图像中不给出任何类标签
##    n_labels = len(set(labels.flat)) 
##    
##    print("图像中没有标签")
##    print(n_labels)
    
    
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1]*original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # C将地图(标签)转换回相应的颜色并保存图像。
    # 注意，这里不再有“未知”，不管我们一开始拥有什么。
    MAP = colorize[MAP,:]
    MAP = MAP.reshape(original_image.shape)
    MAP = array_to_img(MAP)
    MAP.save(output_image)
    return MAP

def crf_PIL(original_image, annotated_image,output_image, use_2d = True):

    annotated_labels = annotated_image.flatten()
    colors,labels = np.unique(annotated_labels, return_inverse=True)
    print(len(labels))
    print(labels)
    print(len(colors))
    print(colors)
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], len(colors))

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, len(colors), gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    print(MAP.shape)

    # C将地图(标签)转换回相应的颜色并保存图像。
    # 注意，这里不再有“未知”，不管我们一开始拥有什么。
    MAP = MAP.reshape(original_image.shape[1],original_image.shape[0],1)
    cv2.imwrite(output_image,MAP)
    #MAP = array_to_img(MAP)
    #MAP.save(output_image)
    return MAP

if __name__=='__main__':
    image = Image.open('C:\\Users\\Admin\\Desktop\\1.tif')
    image = image.convert("RGB")
    annotated_image=Image.open('C:\\Users\\Admin\\Desktop\\2.tif')
    image=np.array(image)
    annotated_image=np.array(annotated_image)
    output = crf_PIL(image,annotated_image,"crf.png")
