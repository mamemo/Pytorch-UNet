from __future__ import print_function

import os
import time
import numpy as np

from unet import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

#from skimage.transform import resize
from skimage.io import imsave, imread
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--preddir", required=True, help="Predictions output folder")
args = vars(ap.parse_args())
pred_dir = args["preddir"]

# K.set_image_data_format('channels_last')

img_rows = 256
img_cols = 256

smooth = 1.

def dice_coef(y_pred, y_true):
    smooth = 1
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
  
def dice_coef_loss(y_pred, y_true):
    return -dice_coef(y_pred, y_true)


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask_test = np.load('imgs_mask_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_mask_test, imgs_id


def get_unet():
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the model
    net = UNet(n_channels=1, n_classes=1).to(device)
    # Definition of the optimizer
    optimizer = optim.Adam(net.parameters(),
                           lr=1e-4)

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = imgs[i]
        #imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    start_time = time.time()
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=64, epochs=30, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])
    print('-'*30)
    print('Trainig time in seconds: %s' % (time.time() - start_time))
    print('-'*30)
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_mask_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_mask_test = preprocess(imgs_mask_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_predict = model.predict(imgs_test, verbose=0)
    np.save('imgs_mask_predict.npy', imgs_mask_predict)
    scores = model.evaluate(imgs_test, imgs_mask_test, verbose=1)
    print("%s: %.2f" % (model.metrics_names[1], scores[1]))

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image_pred, image_id in zip(imgs_mask_predict, imgs_id_test):
        image_pred = ((image_pred[:, :, 0] * 255.) > 127.5).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image_pred)

if __name__ == '__main__':
    train_and_predict()