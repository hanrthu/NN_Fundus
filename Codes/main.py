import torchvision
from torchvision import transforms
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from data_process import load_data, myDataset
from model import resnet, inception, SEnet

import argparse
import os
import logging
import time

parser = argparse.ArgumentParser(description='resnet')
parser.add_argument('--expname', type=str, default="null", help='Experiment name.')
parser.add_argument('--model', default='resnet50', choices=['resnet18', 'resnet50', 'inception', 'senet18', 'senet50'], help="model name")
parser.add_argument('--batchsize', default=32, type=int, help='batch size when training')
parser.add_argument('--gpu', default=0, type=int, help='gpu id, -1 means using cpu')
parser.add_argument('--weight_decay', default=0.0, type=float, help='L2 regularizer coeficient')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--start_epoch', default=0, type=int, help='start_epoch')
parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--dataset_dir', type=str, default='../fundus_hbp', help='Dataset directory')
parser.add_argument('--loadnpy', action="store_true", default=False, help='load from existing npy file')
parser.add_argument('--npydir', type=str, default='baseline', help='npy file directory')
parser.add_argument('-c', '--continue_train', action="store_true", default=False, help='continue to train')
parser.add_argument('-n', '--normalization', action="store_true", default=False, help='use subtractive normalization technique')
parser.add_argument('-a', '--augmentation', action="store_true", default=False, help='use data augmentation')
parser.add_argument('--no_pretrain', action="store_true", default=False, help='do not use pretrained weights')
parser.add_argument('--stack', action="store_true", default=False, help='stack both eyes into one image of 6 channels')
parser.add_argument('--resize_shape', default=224, type=int, help='input image shape, -1 means using original shape')


args = parser.parse_args()
curtime = time.strftime('%Y_%m%d_%H%M%S', time.localtime(time.time()))
if args.expname == "null":
    args.expname = curtime
if not os.path.exists("exps"):
    os.mkdir("exps")
if not os.path.exists("npys"):
    os.mkdir("npys")
if not os.path.exists(os.path.join("exps", args.expname)):
    os.mkdir(os.path.join("exps", args.expname))
if not os.path.exists(os.path.join("npys", args.npydir)):
    os.mkdir(os.path.join("npys", args.npydir))

# set logger
logpath = os.path.join("exps", args.expname, "log.txt")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(logpath, mode='a')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parselog(args, logger):
    for k, v in vars(args).items():
        logger.info('%s: %s' % (k, str(v)))


def calc_acc(predictions, labels):
    result = torch.max(predictions, 1)[1]
    corrects = (result.data == labels.data).sum()
    acc = float(corrects)/labels.shape[0]
    return acc

def train_epoch(model, dataloader, optimizer, device):
    losses = []
    accs = []
    aucs = []
    for i, batch in enumerate(dataloader):
        X_batch, Y_batch = batch
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = F.cross_entropy(output, Y_batch)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.cpu().detach())
        proba = np.array(F.softmax(output, dim=-1)[:, 1].cpu().detach())
        model.eval()
        acc = calc_acc(output, Y_batch)
        try:
            auc = roc_auc_score(Y_batch.cpu(), proba)
            accs.append(acc)
            aucs.append(auc)
        except ValueError:
            pass
        model.train()

    return np.array(losses).mean(), np.array(accs).mean(), np.array(aucs).mean()

def valid_epoch(model, dataloader, device):
    model.eval()
    losses = []
    accs = []
    aucs = []
    for i, batch in enumerate(dataloader):
        X_batch, Y_batch = batch
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        output = model(X_batch)
        loss = F.cross_entropy(output, Y_batch)
        
        losses.append(loss.cpu().detach())
        proba = np.array(F.softmax(output, dim=-1)[:, 1].cpu().detach())
        acc = calc_acc(output, Y_batch)
        try:
            auc = roc_auc_score(Y_batch.cpu(), proba)
            accs.append(acc)
            aucs.append(auc)
        except ValueError:
            pass

    model.train()    
    return np.array(losses).mean(), np.array(accs).mean(), np.array(aucs).mean()


if __name__ == '__main__':
    parselog(args, logger)
    logger.debug("Start...")
    setup_seed(7)
    classnum = 2
    if args.gpu >= 0:
        device = torch.device("cuda:%d"%args.gpu)
    else:
        device = torch.device("cpu")

    # they are all np.array
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(args.dataset_dir, normalization=args.normalization, resize=args.resize_shape, loadnpy=args.loadnpy, npydir=os.path.join("npys", args.npydir), stack=args.stack, logger=logger)

    logger.debug(X_train.shape)
    
    if args.model.startswith("resnet"):
        model = resnet(classnum=classnum, name=args.model, pretrain=not args.no_pretrain, stack=args.stack)
    elif args.model.startswith("senet"):
        model = SEnet(classnum=classnum, name=args.model, pretrain=not args.no_pretrain, stack=args.stack)
    elif args.model.startswith("inception"):
        model = inception(classnum=classnum, pretrain=not args.no_pretrain)
    else:
        assert False

    logger.debug(device)
    model.to(device)

    if args.continue_train:
        model = torch.load(os.path.join("exps", args.expname, "checkpoint_latest.pth.tar"), map_location=lambda storage, loc: storage.cuda(device))
        
    train_data = myDataset(X_train, Y_train, dotrans = args.augmentation, stack=args.stack)    
    trainloader = DataLoader(train_data, batch_size = args.batchsize, shuffle = True)
    val_data = myDataset(X_val, Y_val, dotrans = False, stack=args.stack)
    valloader = DataLoader(val_data, batch_size = args.batchsize, shuffle = False)    
    test_data = myDataset(X_test, Y_test, dotrans = False, stack=args.stack)
    testloader = DataLoader(test_data, batch_size = args.batchsize, shuffle = False)    

    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)    

    logger.debug("Start Training...")
   
    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(args.start_epoch+1, args.start_epoch+args.num_epochs+1):
        start_time = time.time()
        train_loss, train_acc, train_auc = train_epoch(model, trainloader, optimizer, device)
        val_loss, val_acc, val_auc = valid_epoch(model, valloader, device)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            test_loss, test_acc, test_auc = valid_epoch(model, testloader, device)
            with open(os.path.join("exps", args.expname, 'checkpoint_best.pth.tar'), 'wb') as fout:
                torch.save(model, fout)
        with open(os.path.join("exps", args.expname, 'checkpoint_latest.pth.tar'), 'wb') as fout:
            torch.save(model, fout)
        epoch_time = time.time() - start_time

        logger.info("Epoch %d of %d took %.4fs"%(epoch, args.num_epochs, epoch_time))
        logger.info("  training loss:             %.4f" %(train_loss))
        logger.info("  training accuracy:         %.4f" %(train_acc))
        logger.info("  training auc:              %.4f" %(train_auc))
        logger.info("  validation loss:           %.4f" %(val_loss))
        logger.info("  validation accuracy:       %.4f" %(val_acc))
        logger.info("  validation auc:            %.4f" %(val_auc))        
        logger.info("  best epoch:                  %d" %(best_epoch))
        logger.info("  best validation accuracy:  %.4f" %(best_val_acc))
        logger.info("  test loss:                 %.4f" %(test_loss))
        logger.info("  test accuracy:             %.4f" %(test_acc))
        logger.info("  test auc:                  %.4f" %(test_auc))
