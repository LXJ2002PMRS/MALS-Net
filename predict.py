import glob
import numpy as np
import torch
from osgeo import gdal
import network
from collections import OrderedDict
from tqdm import tqdm
import os
gdal.DontUseExceptions()

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "The file unable to open")

    tif_XSize = dataset.RasterXSize
    tif_YSize = dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    tif_data = band.ReadAsArray(0, 0, tif_XSize, tif_YSize)
    tif_data_np = np.array(tif_data)
    del dataset
    return tif_data_np

def writeTiff(fileName, data, im_geotrans=(0, 0, 0, 0, 0, 0), im_proj=""):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        im_bands, im_height, im_width = data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fileName, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(data[i])
    del dataset

if __name__ == "__main__":

    test_image_data_path = glob.glob('../data/test/image\\*.tif')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network.deeplabv3plus_ECAResNet50(
        num_classes=1,
        output_stride=16,
        pretrained_backbone=False
    )
    net = model.to(device=device)

    # 加载模型参数    
    net.load_state_dict(torch.load('utils/check.pth', map_location=device))
    net.eval()
    # 遍历所有图片
    for image_path in tqdm(test_image_data_path):
        feature = OrderedDict()       
        filename = image_path.split('\\')[-1]
        save_dir = 'utils/predict_test/'
        # 检查并创建（如果不存在）
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_res_path = os.path.join(save_dir, filename)
        dem_path = '../data/test/dem/' + image_path.split('\\')[-1]
        image = readTif(image_path)
        dem = readTif(dem_path)
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        dem = dem.reshape(1, 1, dem.shape[0], dem.shape[1])       
        image_tensor = torch.from_numpy(image)
        dem_tensor = torch.from_numpy(dem)
        image = image_tensor.to(device=device, dtype=torch.float32)
        dem = dem_tensor.to(device=device, dtype=torch.float32)
        feature['image'] = image
        feature['dem'] = dem
        pred = net(feature)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        pred = pred.astype(np.uint8)

        writeTiff(save_res_path, pred)
        
