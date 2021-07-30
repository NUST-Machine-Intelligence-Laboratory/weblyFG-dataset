# -*- coding: utf-8 -*

import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from loss_plm import peer_learning_loss
from lr_scheduler import lr_scheduler
from bcnn import BCNN
from resnet import ResNet50
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


torch.manual_seed(0)
torch.cuda.manual_seed(0)

os.popen('mkdir -p model')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--T_k', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. ')
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--step', type=int, default=None)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--n_classes', type=int, default=200)
parser.add_argument('--net1', type=str, default='bcnn',
                    help='specify the network architecture, available options include bcnn, vgg16, vgg19, resnet18, resnet34, resnet50')
parser.add_argument('--net2', type=str, default='bcnn',
                    help='specify the network architecture, available options include bcnn, vgg16, vgg19, resnet18, resnet34, resnet50')

args = parser.parse_args()

data_dir = args.dataset
learning_rate = args.base_lr
batch_size = args.batch_size
num_epochs = args.epoch
drop_rate = args.drop_rate
T_k = args.T_k
weight_decay = args.weight_decay
N_CLASSES = args.n_classes

if args.net1 == 'bcnn':
    NET1 = BCNN
elif args.net1 == 'resnet50':
    NET1 = ResNet50
else:
    raise AssertionError('net should be in bcnn, resnet50')

if args.net2 == 'bcnn':
    NET2 = BCNN
elif args.net2 == 'resnet50':
    NET2 = ResNet50
else:
    raise AssertionError('net should be in bcnn, resnet50')

resume = args.resume

epoch_decay_start = 40
warmup_epochs = 5


logfile = 'logfile_' + data_dir + '_peerlearning_' + str(drop_rate) + '.txt'

# Load data
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(size=448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.CenterCrop(size=448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
train_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = lr_scheduler(learning_rate, num_epochs, warmup_end_epoch=warmup_epochs, mode='cosine')
beta1_plan = [mom1] * num_epochs
for i in range(epoch_decay_start, num_epochs):
    beta1_plan[i] = mom2


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # only change beta1


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

rate_schedule = np.ones(num_epochs) * drop_rate
rate_schedule[:T_k] = np.linspace(0, drop_rate, T_k)


def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    N = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target is in shape (N,)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)  # size is 1
        res.append(correct_k.mul_(100.0 / N))
    return res


# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0
    epoch_loss1 = []
    epoch_loss2 = []
    for it, (images, labels) in enumerate(train_loader):
        # if it > 60:
        #     break
        iter_start_time = time.time()

        images = images.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        logits1 = model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec1.item()

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2 += 1
        train_correct2 += prec2.item()
        loss_1, loss_2 = peer_learning_loss(logits1, logits2, labels, rate_schedule[epoch])
        epoch_loss1.append(loss_1.item())
        epoch_loss2.append(loss_2.item())

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        iter_end_time = time.time()
        print('Epoch:[{0:03d}/{1:03d}]  Iter:[{2:04d}/{3:04d}]  '
              'Train Accuracy 1:[{4:6.2f}]  Train Accuracy 1:[{5:6.2f}]  Loss 1:[{6:4.4f}]  Loss 2:[{7:4.4f}]  '
              'Iter Runtime:[{8:6.2f}]'.format(
               epoch+1, num_epochs, it+1, len(train_data)//batch_size,
               prec1.item(), prec2.item(), loss_1.item(), loss_2.item(),
               iter_end_time - iter_start_time))

    train_acc1 = float(train_correct) / float(train_total)
    train_acc2 = float(train_correct2) / float(train_total2)
    return train_acc1, train_acc2


def evaluate(test_loader, model1, model2):
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels in test_loader:
        images = images.cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum().item()

    model2.eval()  # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for images, labels in test_loader:
        images = images.cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum().item()

    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)
    return acc1, acc2


def main():
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    step = args.step
    print('===> About training in a two-step process! ===')
    if step == 1:
        print('===> Step 1 ...')
        cot1 = NET1(n_classes=N_CLASSES, pretrained=True, use_two_step=True)
        cot1 = nn.DataParallel(cot1).cuda()
        optimizer1 = optim.Adam(cot1.module.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
        cot2 = NET2(n_classes=N_CLASSES, pretrained=True, use_two_step=True)
        cot2 = nn.DataParallel(cot2).cuda()
        optimizer2 = optim.Adam(cot2.module.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif step == 2:
        print('===> Step 2 ...')
        cot1 = NET1(n_classes=N_CLASSES, pretrained=False, use_two_step=True)
        cot1 = nn.DataParallel(cot1).cuda()
        optimizer1 = optim.Adam(cot1.parameters(), lr=learning_rate, weight_decay=weight_decay)
        cot2 = NET2(n_classes=N_CLASSES, pretrained=False, use_two_step=True)
        cot2 = nn.DataParallel(cot2).cuda()
        optimizer2 = optim.Adam(cot2.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif step == 0:
        print('===> One Step Training ...')
        cot1 = NET1(n_classes=N_CLASSES, pretrained=False, use_two_step=False)
        cot1 = nn.DataParallel(cot1).cuda()
        optimizer1 = optim.Adam(cot1.parameters(), lr=learning_rate, weight_decay=weight_decay)
        cot2 = NET2(n_classes=N_CLASSES, pretrained=False, use_two_step=False)
        cot2 = nn.DataParallel(cot2).cuda()
        optimizer2 = optim.Adam(cot2.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise AssertionError('Wrong step argument')

    # check if it is resume mode
    print('-----------------------------------------------------------------------------')
    if resume:
        assert os.path.isfile('checkpoint.pth'), 'please make sure checkpoint.pth exists'
        print('---> loading checkpoint.pth <---')
        checkpoint = torch.load('checkpoint.pth')
        assert step == checkpoint['step'], 'step in checkpoint does not match step in argument'
        start_epoch = checkpoint['epoch']
        best_accuracy1 = checkpoint['best_accuracy1']
        best_accuracy2 = checkpoint['best_accuracy2']
        best_epoch1 = checkpoint['best_epoch1']
        best_epoch2 = checkpoint['best_epoch2']
        cot1.load_state_dict(checkpoint['cot1_state_dict'])
        cot2.load_state_dict(checkpoint['cot2_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
    else:
        print('--->        no checkpoint loaded         <---')
        if step == 2:
            cot1.load_state_dict(torch.load('model/net1_step1_vgg16_best_epoch.pth'))
            cot2.load_state_dict(torch.load('model/net2_step1_vgg16_best_epoch.pth'))
        start_epoch = 0
        best_accuracy1, best_accuracy2 = 0.0, 0.0
        best_epoch1, best_epoch2 = None, None
    print('-----------------------------------------------------------------------------')

    with open(logfile, "a") as f:
        f.write('------ Step: {} ...\n'.format(step))

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        cot1.train()
        adjust_learning_rate(optimizer1, epoch)
        cot2.train()
        adjust_learning_rate(optimizer2, epoch)

        train_acc1, train_acc2 = train(train_loader, epoch, cot1, optimizer1, cot2, optimizer2)
        test_acc1, test_acc2 = evaluate(test_loader, cot1, cot2)

        if test_acc1 > best_accuracy1:
            best_accuracy1 = test_acc1
            best_epoch1 = epoch + 1
            torch.save(cot1.state_dict(), 'model/net1_step{}_vgg16_best_epoch.pth'.format(step))
        if test_acc2 > best_accuracy2:
            best_accuracy2 = test_acc2
            best_epoch2 = epoch + 1
            torch.save(cot2.state_dict(), 'model/net2_step{}_vgg16_best_epoch.pth'.format(step))
        epoch_end_time = time.time()
        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'cot1_state_dict': cot1.state_dict(),
            'cot2_state_dict': cot2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'best_epoch1': best_epoch1,
            'best_accuracy1': best_accuracy1,
            'best_epoch2': best_epoch2,
            'best_accuracy2': best_accuracy2,
            'step': step,
        })

        print('------\n'
              'Epoch: [{:03d}/{:03d}]\tTrain Accuracy 1: [{:6.2f}]\tTrain Accuracy 2: [{:6.2f}]\t'
              'Test Accuracy 1: [{:6.2f}]\tTest Accuracy 2: [{:6.2f}]\t'
              'Epoch Runtime: [{:6.2f}]'
              '\n------'.format(
               epoch + 1, num_epochs, train_acc1, train_acc2, test_acc1, test_acc2,
               epoch_end_time - epoch_start_time))
        with open(logfile, "a") as f:
            output = 'Epoch: [{:03d}/{:03d}]\tTrain Accuracy 1: [{:6.2f}]\tTrain Accuracy 1: [{:6.2f}]\t' \
                     'Test Accuracy 1: [{:6.2f}]\tTest Accuracy 2: [{:6.2f}]\t' \
                     'Epoch Runtime: [{:6.2f}]'.format(
                      epoch + 1, num_epochs, train_acc1, train_acc2, test_acc1, test_acc2,
                      epoch_end_time - epoch_start_time)
            f.write(output + "\n")

    print('******\n'
          'Best Accuracy 1: [{0:6.2f}], at Epoch [{1:03d}]; '
          'Best Accuracy 2: [{2:6.2f}], at Epoch [{3:03d}].'
          '\n******'.format(best_accuracy1, best_epoch1, best_accuracy2, best_epoch2))
    with open(logfile, "a") as f:
        output = '******\n' \
                 'Best Accuracy 1: [{0:6.2f}], at Epoch [{1:03d}]; ' \
                 'Best Accuracy 2: [{2:6.2f}], at Epoch [{3:03d}].' \
                 '\n******'.format(best_accuracy1, best_epoch1, best_accuracy2, best_epoch2)
        f.write(output + "\n")


if __name__ == '__main__':
    main()
