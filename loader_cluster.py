from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random
import re


class BBBCDataset(Dataset):
    def __init__(self, ids_data, ids_gt, direc):

        self.direc = direc

        # Transforms
        self.transformations = transforms.ToTensor()

        # Images IDS
        self.ids_data = ids_data
        self.ids_gt = ids_gt

        # Calculate len of data
        self.data_len = len(self.ids_data)

    def __getitem__(self, index):
        # Get an ID of a specific image
        id_img = self.direc + self.ids_data[index]
        id_gt = self.direc + self.ids_gt[index]
        # Open Image and GroundTruth
        img = Image.open(id_img)
        gt = Image.open(id_gt)
        # Applies transformations
        img = self.transformations(img)
        gt = self.transformations(gt)

        return (img, gt)

    def __len__(self):
        return self.data_len


def create_dataset(direc):
    images = os.listdir(direc)
    ids_data = []
    ids_gt = []
    for image_name in images:
        if 'groundtruth' in image_name:
            continue
        img_id = re.sub("_original","", image_name,count=1)
        image_mask_name = '_groundtruth_(1)_' + img_id
        ids_data.append(image_name)
        ids_gt.append(image_mask_name)
    
    dataset = BBBCDataset(ids_data=ids_data, ids_gt=ids_gt, direc=direc)
    return dataset

def get_dataloaders(dir_train, dir_test, batch_size=10):
    
    # Create the datasets
    train_dataset = create_dataset(dir_train)
    test_dataset = create_dataset(dir_test)

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_predloader(dir_img, dir_gt, batch_size=1):
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Rearrange the images
    random.shuffle(ids)
    # Calculate index of partition
    ids_pred = ids[:10]

    # Create the datasets
    pred_dataset = BBBCDataset(ids=ids_pred, dir_data=dir_img, dir_gt=dir_gt)

    # Create the loaders
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=True)

    return pred_loader
