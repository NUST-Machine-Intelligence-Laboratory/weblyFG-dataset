# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class BCNN(nn.Module):
    """
    BCNN

    The structure of BCNN is as follows:
        conv1_1 (64) -> relu -> conv1_2 (64) -> relu -> pool1(64*224*224)
    ->  conv2_1(128) -> relu -> conv2_2(128) -> relu -> pool2(128*112*112)
    ->  conv3_1(256) -> relu -> conv3_2(256) -> relu -> conv3_3(256) -> relu -> pool3(256*56*56)
    ->  conv4_1(512) -> relu -> conv4_2(512) -> relu -> conv4_3(512) -> relu -> pool4(512*28*28)
    ->  conv5_1(512) -> relu -> conv5_2(512) -> relu -> conv5_3(512) -> relu(512*28*28)
    ->  bilinear pooling(512**2)
    ->  fc(n_classes)

    The network input 3 * 448 * 448 image
    The output of last convolution layer is 512 * 14 * 14

    Extends:
        torch.nn.Module
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=True):
        super().__init__()
        print('| A BCNN network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1])
        self.fc = nn.Linear(512**2, self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

            if use_two_step:
                # Freeze all layer in self.feature
                for params in self.features.parameters():
                    params.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the network

        Arguments:
            x [torch.Tensor] -- shape is (N, 3, 448, 448)

        Return:
            x [torch.Tensor] -- shape is (N, 200)
        """
        N = x.size(0)
        # assert x.size() == (N, 3, 448, 448), 'The image size should be 3 x 448 x 448'
        x = self.features(x)
        bp_output = self.bilinear_pool(x)
        x = self.fc(bp_output)
        assert x.size() == (N, self._n_classes)
        return x

    @staticmethod
    def bilinear_pool(x):
        N, ch, h, w = x.shape
        # assert x.size() == (N, 512, 28, 28)
        x = x.view(N, 512, h*w)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (h * w)
        x = x.view(N, 512**2)
        x = torch.sqrt(x + 1e-5)
        x = F.normalize(x)
        assert x.size() == (N, 512**2)
        return x
