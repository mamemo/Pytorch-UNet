
from __future__ import print_function

import os
import time
import numpy as np

from data import load_train_data, load_test_data
from unet_model import unet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as utils

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


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = imgs[i]
        #imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_net(net, device, loader, dir_checkpoint,optimizer, epochs, run=""):
    ''' Train the CNN. '''
    for epoch in range(epochs):
        print('\nStarting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()
        train_loss = 0
        cont = 0
        for batch_idx, (data, gt) in enumerate(loader):

            # Use GPU or not
            data, gt = data.to(device, dtype=torch.float), gt.to(device, dtype=torch.float)

            optimizer.zero_grad()

            # Forward
            predictions = net(data)

            # To calculate Loss
            pred_probs = torch.sigmoid(predictions)

            # Loss Calculation
            train_loss += dice_coef_loss(pred_probs, gt).item()
            cont += 1

            # Backpropagation
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))

        train_loss /= cont
        print('\nAverage Training Loss: ' + str(train_loss))
    
    # Save the weights
    torch.save(net.state_dict(), dir_checkpoint + 'weights'+run+'.pth')
        
    return train_loss

def test_net(net, device, loader):
    ''' Test the CNN '''
    net.eval()
    test_loss = 0
    cont = 0
    with torch.no_grad():
        for data, gt in loader:

            # Use GPU or not
            data, gt = data.to(device, dtype=torch.float), gt.to(device, dtype=torch.float)

            # Forward
            predictions = net(data)

            # To calculate Loss
            pred_probs = torch.sigmoid(predictions)
            
            # Loss Calculation
            test_loss += dice_coef(pred_probs, gt).item()
            cont += 1

    test_loss /= cont
    print('\nTest set: Average loss: '+ str(test_loss))
    return test_loss


def train_and_test():
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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

    tensor_x = torch.stack([torch.Tensor(i) for i in imgs_train]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in imgs_mask_train])

    train_dataloader = utils.DataLoader(utils.TensorDataset(tensor_x,tensor_y), batch_size=64, shuffle=True) #create dataset and dataloader

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = unet(n_classes=1, in_channels=1)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    train_net(model, device, train_dataloader, 'checkpoints/',optimizer, epochs=30)
    
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

    tensor_x = torch.stack([torch.Tensor(i) for i in imgs_test]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in imgs_mask_test])

    test_dataloader = utils.DataLoader(utils.TensorDataset(tensor_x,tensor_y), batch_size=64, shuffle=True) #create dataset and dataloader

    # print('-'*30)
    # print('Loading saved weights...')
    # print('-'*30)
    # model.load_weights('weights.h5')

    print('-'*30)
    print('Evaluating model...')
    print('-'*30)
    
    test_net(model, device, test_dataloader)

    # print('-' * 30)
    # print('Saving predicted masks to files...')
    # print('-' * 30)
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # for image_pred, image_id in zip(imgs_mask_predict, imgs_id_test):
    #     image_pred = ((image_pred[:, :, 0] * 255.) > 127.5).astype(np.uint8)
    #     imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image_pred)

if __name__ == '__main__':
    train_and_test()
    
