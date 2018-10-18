import sys
import os
from optparse import OptionParser
import numpy as np
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from unet import UNet
from loader_cluster import load_train_data, load_test_data, preprocess


def time_me(*arg):
    if len(arg) != 0: 
        elapsedTime = time.time() - arg[0]
        hours = math.floor(elapsedTime / (60*60))
        elapsedTime = elapsedTime - hours * (60*60)
        minutes = math.floor(elapsedTime / 60)
        elapsedTime = elapsedTime - minutes * (60)
        seconds = math.floor(elapsedTime)
        elapsedTime = elapsedTime - seconds
        ms = elapsedTime * 1000
        if(hours != 0):
            return "%d hours %d minutes %d seconds" % (hours, minutes, seconds)
        elif(minutes != 0):
            return "%d minutes %d seconds" % (minutes, seconds)
        else :
            return "%d seconds %f ms" % (seconds, ms)
    else:
        return time.time()


def dice_coef(y_pred, y_true):
    smooth = 1
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
  
def dice_coef_loss(y_pred, y_true):
    return -dice_coef(y_pred, y_true)

def train_net(net, device, data_train, gt_train, amount, dir_checkpoint,optimizer, epochs=5, run=""):
    ''' Train the CNN. '''
    for epoch in range(epochs):
        print('\nStarting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()
        train_loss = 0
        cont = 0
        time_var = time_me()
        for idx in range(amount):

            # Use GPU or not
            data, gt = data_train[idx].to(device, dtype=torch.float), gt_train[idx].to(device, dtype=torch.float)

            optimizer.zero_grad()

            # Forward
            predictions = net(data)

            # To calculate Loss
            pred_probs = torch.sigmoid(predictions)
            # pred_probs_flat = pred_probs.view(-1)
            # gt_flat = gt.view(-1)

            # Loss Calculation
            loss = dice_coef_loss(pred_probs, gt)
            train_loss += loss.item()
            cont += 1

            # Backpropagation
            loss.backward()
            optimizer.step()

            if idx%10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, idx, amount, 100. * idx / amount, loss.item()))

        train_loss /= cont
        print('\nAverage Training Loss: ' + str(train_loss))
        print('Train Time: It tooks '+time_me(time_var)+' to finish the epoch.')
    
    # Save the weights
    torch.save(net.state_dict(), dir_checkpoint + 'weights'+run+'.pth')
        
    return train_loss

def test_net(net, device, loader):
    ''' Test the CNN '''
    net.eval()
    test_loss = 0
    cont = 0
    time_var = time_me()
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
    print('Test time: It tooks '+time_me(time_var)+' to finish the Test.')
    return test_loss

def setup_and_run_train(load = False, batch_size = 10,
                epochs = 5, lr = 0.1, run="", dir_train="", dir_test=""):
    
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the model
    net = UNet(n_channels=1, n_classes=1).to(device)

    # Load old weights
    if load:
        net.load_state_dict(torch.load(load))
        print('Model loaded from {}'.format(load))

    # Location of the images to use
    dir_checkpoint = 'checkpoints/'
    pred_dir = 'pred/'

    # Load the dataset
    train_loader, test_loader = get_dataloaders(
        dir_train, dir_test, batch_size)

    # Pretty print of the run
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Testing size: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_loader.dataset),
               len(test_loader.dataset), str(use_cuda)))

    # Definition of the optimizer
    optimizer = optim.Adam(net.parameters(),
                           lr=lr)

    # Run the training and testing
    try:
        time_var = time_me()
        train_loss = train_net(net=net,
                  epochs=epochs,
                  device=device,
                  dir_checkpoint=dir_checkpoint,
                  loader=train_loader,
                  optimizer=optimizer,
                  run = run)
        test_loss = test_net(net=net, device=device, loader=test_loader)
        print('\nRun time: It tooks '+time_me(time_var)+' to finish the run.')
        return train_loss, test_loss
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=30, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-r', '--runs', dest='runs',
                      default=10, help='How many runs')                  

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    setup_and_run_train(load=args.load,
                        batch_size=args.batchsize,
                        epochs=args.epochs,
                        lr=args.lr
                        )

    # runs = args.runs
    # acum_train = 0
    # acum_test = 0
    # for i in range(1,runs+1):
    #     print('-'*10 + 'Start run {}'.format(i) + '-'*10)
    #     train_loss, test_loss = setup_and_run_train(load = args.load,
    #             batch_size = args.batchsize,
    #             epochs = args.epochs,
    #             lr = args.lr, run=str(i),
    #             dir_train='/home/scalderon/unet/raw/hoechst/original/train_'+str(i)+'/output/', 
    #             dir_test='/home/scalderon/unet/raw/hoechst/original/test_'+str(i)+'/output/')
    #     acum_train += train_loss
    #     acum_test += test_loss

    # acum_train /= runs
    # acum_test /= runs

    # print('\nAfter '+str(runs)+' runs: \n\tAverage Train Loss: '+str(acum_train)+'\n\tAverage Test Loss: '+str(acum_test))
