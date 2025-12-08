import torch
import os
import glob
from torch.utils.data import Dataset
from osgeo import gdal
import numpy as np
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

class MoonDetection3(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        dem_path = image_path.replace('image', 'dem')
        label_path = image_path.replace('image', 'label')
        image = readTif(image_path)
        dem = readTif(dem_path)       
        label = readTif(label_path)
        # Turn the data into a single-channel
        image = image.reshape(1, image.shape[0], image.shape[1])
        dem = dem.reshape(1, dem.shape[0], dem.shape[1])        
        label = label.reshape(1, label.shape[0], label.shape[1])
        image= image / 1.0
        dem= dem/1.0        
        label=label/1.0
        if label.max() > 1:
            label = label / 255
        return image, dem,  label

    def __len__(self):
        return len(self.imgs_path)

if __name__=="__main__":
    ismoon_dataset = MoonDetection3('../data/')
    print("Total number of data：", len(ismoon_dataset))
    train_loder = torch.utils.data.DataLoader(dataset=ismoon_dataset, batch_size=16, shuffle=True)
    for image, dem,  label in train_loder:
        print(image.shape)







