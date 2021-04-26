import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.vit import ViT, channel_selection

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

model = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,                  # 512
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    )
model = model.to(device)

model_path = 'checkpoint/pruning-adamw-vit-4-79.72.t7'
print("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['acc']
model.load_state_dict(checkpoint['net'])
print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))

total = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        total += m.indexes.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        size = m.indexes.data.shape[0]
        bn[index:(index+size)] = m.indexes.data.abs().clone()
        index += size

percent = 0.3
y, i = torch.sort(bn)
thre_index = int(total * percent)
thre = y[thre_index]

# print(thre)
pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, channel_selection):
        weight_copy = m.indexes.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.indexes.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total
print('Pre-processing Successful!')
print(cfg)

def test(model):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='/home/lxc/ABCPruner/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

# test(model)
