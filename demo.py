# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from bcnn import BCNN
from resnet import ResNet50
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str , required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_classes', type=int, default=200)
    parser.add_argument('--net', type=str, default='bcnn')
    args = parser.parse_args()

    data_dir = args.data
    model_path = args.model
    N_CLASSES = args.n_classes

    if args.net == 'bcnn':
        NET = BCNN
    elif args.net == 'resnet50':
        NET = ResNet50
    else:
        raise AssertionError('net should be in bcnn, resnet50')

    net = NET(n_classes=N_CLASSES, pretrained=False)

    if torch.cuda.device_count() >= 1:
        net = torch.nn.DataParallel(net).cuda()
        print('cuda device : ', torch.cuda.device_count())
    else:
        raise EnvironmentError('This is designed to run on GPU but no GPU is found')
    net.load_state_dict(torch.load(model_path))

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.CenterCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    net.eval()
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for X, y in test_loader:
            # Data
            X = X.cuda()
            y = y.cuda(async=True)
            # Prediction
            score = net(X)
            _, prediction = torch.max(score, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data).item()

    test_accuracy = 100 * num_correct / num_total

    print('-----------------------------------------------------------------')
    print('Test accuracy: {}'.format(test_accuracy))
    print('-----------------------------------------------------------------')
