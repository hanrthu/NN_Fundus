import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import albumentations
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

def mask(img,mask_type,mask_ratio):
    height = img.shape[0]
    width = img.shape[1]
    if mask_type==1:
        lower_limit = int(height * mask_ratio // 2 )
        upper_limit = height - lower_limit
        img[:lower_limit,:] = [0,0,0]
        img[upper_limit:,:] = [0,0,0]
    elif mask_type==2:
        img1 = img.copy()
        img2 = mask(img,4,1-mask_ratio)
        img = cv2.subtract(img1, img2, dst=None, mask=None)
        cv2.imwrite("example_type_"+str(mask_type)+"_"+str(mask_ratio)+".jpg",img)
        exit(0)
    elif mask_type==3:
        img1 = img.copy()
        img2 = mask(img,1,1-mask_ratio)
        img = cv2.subtract(img1, img2, dst=None, mask=None)
        cv2.imwrite("example_type_"+str(mask_type)+"_"+str(mask_ratio)+".jpg",img)
        exit(0)
    elif mask_type==4:
        full_radius = min(height,width)//2
        assert mask_ratio > 0
        radius = int(math.sqrt(mask_ratio)*full_radius)
        img = cv2.circle(img,(width//2,height//2),radius,(0),-1)
        cv2.imwrite("example_type_"+str(mask_type)+"_"+str(mask_ratio)+".jpg",img)
        exit(0)
    else:
        print("error! mask type out of range")
        exit(-1) 
    return img

def load_data(dataset_dir, split_ratio = [0.8, 0.1, 0.1], normalization = False, resize=-1, loadnpy = True, npydir="baseline", stack=False, logger=None,mask_type=0, mask_ratio=0.0):
    logger.debug("Start to read in data...")
    if loadnpy:
        X = np.load(os.path.join(npydir, "X.npy"))
        Y = np.load(os.path.join(npydir, "Y.npy"))
    else:
        img_dir = os.path.join(dataset_dir, "images")
        label_path = os.path.join(dataset_dir, "data.csv")
        img_paths = []
        Y = []
        with open(label_path) as f:
            for line in f:
                content = line.split(',')
                if content[0] == 'image_path':
                    continue
                img_paths.append(content[0])
                Y.append(int(content[2].strip('\n')))
    
        X = []
        cnt = 0
        for img_path in img_paths:
            path = os.path.join(img_dir, img_path)
            img = cv2.imread(path)
            if mask_type!=0:
                img = mask(img,mask_type,mask_ratio)
            if resize != -1:
                img = cv2.resize(img, (resize, resize))
            if normalization:
                img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
            cnt += 1
            if cnt % 100 == 0:
                logger.debug("%d/%d have been read in" % (cnt, len(Y)))
            X.append(img)

        stacked_X = []
        stacked_Y = []
        if stack:
            assert len(X)%2 == 0
            assert len(Y)%2 == 0
            for i in range(0, len(X), 2):
                stacked_X.append(np.concatenate((X[i], X[i+1]), axis=-1))
                assert Y[i] == Y[i+1]
                stacked_Y.append(Y[i])
            X = np.array(stacked_X)
            Y = np.array(stacked_Y)
        else:
            X = np.array(X)
            Y = np.array(Y)
        
        np.save(os.path.join(npydir, "X.npy"), X)
        np.save(os.path.join(npydir, "Y.npy"), Y)

    logger.debug('Done!')
    num_samples = len(X)
    logger.debug("Number of samples: %d" % num_samples)
    print(X.shape)
    print(Y.shape)
    assert sum(split_ratio) == 1
    X_train = X[:int(num_samples*split_ratio[0])]
    X_val = X[int(num_samples*split_ratio[0]): int(num_samples*(split_ratio[0]+split_ratio[1]))]
    X_test = X[int(num_samples*(split_ratio[0]+split_ratio[1])):]
    Y_train = Y[:int(num_samples*split_ratio[0])]
    Y_val = Y[int(num_samples*split_ratio[0]): int(num_samples*(split_ratio[0]+split_ratio[1]))]
    Y_test = Y[int(num_samples*(split_ratio[0]+split_ratio[1])):]
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


class myDataset(Dataset):    
    def __init__(self, X, Y, dotrans = True, stack = False):
        self.X = X
        self.Y = Y
        '''
        self.transform = transforms.Compose([transforms.ToPILImage(),
                              transforms.RandomRotation((30,150)), # 随机旋转30至150度
                              transforms.RandomHorizontalFlip(0.6),# 水平翻转
                              transforms.RandomVerticalFlip(0.4),  # 垂直翻转
                              transforms.ToTensor()])              #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]，而且会将[w,h,c]转化成pytorch需要的[c,w,h]格式
        '''
        self.album = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.8),
            albumentations.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, p=0.5),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
            ],p=0.1),
            albumentations.GaussianBlur(blur_limit=5, p=0.1),
            albumentations.OpticalDistortion(p=0.3),
            albumentations.GaussNoise(p=0.3),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            albumentations.RGBShift(p=0.5),
        ])
        self.notransform = transforms.Compose([transforms.ToPILImage(),
                              transforms.ToTensor()])  
        self.dotrans = dotrans
        self.stack = stack

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.dotrans:
            if self.stack:
                right_eye = self.notransform(self.album(image=self.X[idx, :, :, :3])["image"]) 
                left_eye  = self.notransform(self.album(image=self.X[idx, :, :, 3:])["image"])
                return torch.cat((right_eye, left_eye)), torch.tensor(self.Y[idx])
            else:
                return self.notransform(self.album(image=self.X[idx])["image"]), torch.tensor(self.Y[idx])
            ##return self.transform(self.X[idx]), self.Y[idx]
        else:
            if self.stack:
                right_eye = self.notransform(self.X[idx, :, :, :3]) 
                left_eye  = self.notransform(self.X[idx, :, :, 3:])
                return torch.cat((right_eye, left_eye)), torch.tensor(self.Y[idx])
            else:
                return self.notransform(self.X[idx]), torch.tensor(self.Y[idx])