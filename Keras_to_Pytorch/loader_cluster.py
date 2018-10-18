import os
import numpy as np
import re
from skimage.io import imsave, imread
import argparse

def create_train_data(train_data_path, image_rows, image_cols):
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'groundtruth' in image_name:
            continue
        img_id = re.sub("_original","", image_name,count=1)
        image_mask_name = '_groundtruth_(1)_' + img_id
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask
        i += 1
        if i == total:
            print('Done: {0}/{1} images'.format(i, total))
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = imgs[i]
        #imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def create_test_data(test_data_path, image_rows, image_cols):
    images = os.listdir(test_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=object)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        if 'groundtruth' in image_name:
            continue
        img_id = re.sub("_original","", image_name,count=1)
        image_mask_name = '_groundtruth_(1)_' + img_id
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(test_data_path, image_mask_name), as_gray=True)
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_id[i] = img_id
        imgs_mask[i] = img_mask
        i += 1
        if i == total:
            print('Done: {0}/{1} images'.format(i, total))
    print('Loading done.')
    np.save('imgs_test.npy', imgs)
    np.save('imgs_mask_test.npy', imgs_mask)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask_test = np.load('imgs_mask_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_mask_test, imgs_id


def create_data(train_data_path, test_data_path, image_rows = 256, image_cols = 256, image_ext = 'png'):
    print('-'*60)
    print('Model Data Set:')
    print(train_data_path)
    print(test_data_path)
    print('-'*60)

    create_train_data(train_data_path,image_rows, image_cols)
    create_test_data(test_data_path, image_rows, image_cols)