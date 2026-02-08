import os
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from os.path import join
import cv2
import glob
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class LLIEDataset(torch.utils.data.Dataset):
    def __init__(self, ori_root, lowlight_root, transforms, istrain = False, isdemo = False, dataset_type = 'LOL-v1', cropsize=None):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.matching_dict = {}
        self.file_list = []
        self.istrain = istrain
        self.get_image_pair_list(dataset_type)
        self.transforms = transforms
        self.isdemo = isdemo
        self.cropsize = cropsize
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, ll_image_name = self.file_list[item]
        ori_image = self.transforms(
            Image.open(ori_image_name).convert('RGB')
            )
        LL_image = self.transforms(
            Image.open(ll_image_name).convert('RGB')
            )
        
        if self.istrain:
            ori_image, LL_image = self.preprocessing(ori_image, LL_image)
        
        if self.isdemo:
            return ori_image, LL_image, LL_image, ori_image_name.split('/')[-1].split("\\")[-1]
        
        return ori_image, LL_image, LL_image

    def __len__(self):
        return len(self.file_list)
    
    def get_image_pair_list(self, dataset_type):

        if dataset_type == 'LOL-v1':
            image_name_list = [join(self.lowlight_root, x) for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, key), 
                                    os.path.join(self.lowlight_root, key)])
        elif dataset_type in ['LOL-v2', 'LOL-v2-real', 'LOL-v2-Syn']:

            self.lowlight_root=self.ori_root
            
            if self.istrain:
                mode = 'Train'
            else:
                mode = 'Test'

            if dataset_type == 'LOL-v2' and mode == 'Test':
                dataset_type = 'LOL-v2-real'

            if dataset_type in ['LOL-v2-real', 'LOL-v2']:
                Real_Low_root = join(self.lowlight_root, 'Real_captured', mode, "Low")
                Real_High_root = join(self.ori_root, 'Real_captured', mode, "Normal")
            if dataset_type in ['LOL-v2-Syn', 'LOL-v2']:
                Synthetic_Low_root = join(self.lowlight_root, 'Synthetic', mode, "Low")
                Synthetic_High_root = join(self.ori_root, 'Synthetic', mode, "Normal")
            
            # For Real
            if dataset_type == 'LOL-v2-Syn':
                Real_name_list =[]
            else:
                Real_name_list = [join(Real_Low_root, x) for x in os.listdir(Real_Low_root) if is_image_file(x)]
            
            for key in Real_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(Real_High_root, 'normal'+key[3:]), 
                                    os.path.join(Real_Low_root, key)])
            
            # For Synthetic

            if dataset_type == 'LOL-v2-real':
                Synthetic_name_list =[]
            else:
                Synthetic_name_list = [join(Synthetic_Low_root, x) for x in os.listdir(Synthetic_Low_root) if is_image_file(x)]
            
            for key in Synthetic_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(Synthetic_High_root, key), 
                                    os.path.join(Synthetic_Low_root, key)])
        
        elif dataset_type == 'RESIDE':
            image_name_list = [x for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            # if self.istrain:
            if os.path.isfile( os.path.join(self.ori_root, image_name_list[0].split('_')[0]+'.jpg')):
                FileE = '.jpg'
            else:
                FileE = '.png'
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, key.split('_')[0]+FileE), 
                                    os.path.join(self.lowlight_root,key)])   
        elif dataset_type == 'expe':
            image_name_list = [x for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            if os.path.isfile( os.path.join(self.ori_root, '_'.join(image_name_list[0].split('_')[:-1])+'.jpg')):
                FileE = '.jpg'
            else:
                FileE = '.png'
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, '_'.join(key.split('_')[:-1])+FileE), 
                                    os.path.join(self.lowlight_root,key)])   
        elif dataset_type == 'VE-LOL':
            image_name_list = [join(self.lowlight_root, x) for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, key.replace('low', 'normal',)), 
                                    os.path.join(self.lowlight_root, key)])
        else:
            raise ValueError(str(dataset_type) + "does not support! Please change your dataset type")
                
        if self.istrain or (dataset_type[:6] == 'LOL-v2'):
            random.shuffle(self.file_list)

    def add_dataset(self, ori_root, lowlight_root, dataset_type = 'LOL-v1',):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.get_image_pair_list(dataset_type)

    def preprocessing(self, ORI_image, LL_image):
        if self.cropsize is not None:
            # random cropping
            i, j, h, w = transforms.RandomCrop.get_params(ORI_image, output_size=(self.cropsize, self.cropsize))
            ORI_image = ORI_image[:, i:i+h, j:j+w]
            LL_image = LL_image[:, i:i+h, j:j+w]
            
            # Data Augmentations
            # https://github.com/cuiziteng/Illumination-Adaptive-Transformer/blob/main/IAT_enhance/data_loaders/lol_v1_new.py
            aug = random.randint(0, 8)
            if aug == 1:
                LL_image = LL_image.flip(1)
                ORI_image = ORI_image.flip(1)
            elif aug == 2:
                LL_image = LL_image.flip(2)
                ORI_image = ORI_image.flip(2)
            elif aug == 3:
                LL_image = torch.rot90(LL_image, dims=(1, 2))
                ORI_image = torch.rot90(ORI_image, dims=(1, 2))
            elif aug == 4:
                LL_image = torch.rot90(LL_image, dims=(1, 2), k=2)
                ORI_image = torch.rot90(ORI_image, dims=(1, 2), k=2)
            elif aug == 5:
                LL_image = torch.rot90(LL_image, dims=(1, 2), k=3)
                ORI_image = torch.rot90(ORI_image, dims=(1, 2), k=3)
            elif aug == 6:
                LL_image = torch.rot90(LL_image.flip(1), dims=(1, 2))
                ORI_image = torch.rot90(ORI_image.flip(1), dims=(1, 2))
            elif aug == 7:
                LL_image = torch.rot90(LL_image.flip(2), dims=(1, 2))
                ORI_image = torch.rot90(ORI_image.flip(2), dims=(1, 2))
        
        return ORI_image, LL_image
class LLIE_Dataset(LLIEDataset):
    def __init__(self, ori_root, lowlight_root, transforms, istrain = True):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.image_name_list = glob.glob(os.path.join(self.lowlight_root, '*.png'))
        self.matching_dict = {}
        self.file_list = []
        self.istrain = istrain
        self.get_image_pair_list()
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, ll_image_name = self.file_list[item]
        ori_image = self.transforms(
            Image.open(ori_image_name)
            )

        LL_image_PIL = Image.open(ll_image_name)
        LL_image = self.transforms(
            LL_image_PIL
            )
        
        return ori_image, LL_image

    def __len__(self):
        return len(self.file_list)