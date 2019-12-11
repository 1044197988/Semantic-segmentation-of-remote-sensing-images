from osgeo import gdal
import numpy as np

def load_img(path):
    dataset = gdal.Open(path)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)
    im_data = im_data.transpose((1,2,0))
    return im_data
