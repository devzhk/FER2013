from fer import FER2013
from torch.autograd import Variable
from models import *
import torch
import torch.nn as nn
import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


cut_size = 44
bs = 1
model_path = 'FER2013_Resnet18/PublicTest_model.t7'

epls = [0.001 * i for i in range(11)]

transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(cut_size),
    transforms.ToTensor(),
])


def fgsm_attack(img, eps, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_img = img + eps * sign_data_grad
    perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img


def test(model, device, test_loader, eps):
    correct = 0
    adv_egs = []
    for data, target in test_loader:
        # batch_size, ncrops, c, h , w = np.shape(data)
        # data = data.view(-1, c, h, w)

        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, eps, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (eps == 0) and (len(adv_egs) < 5):
                adv_eg = perturbed_data.squeeze().detach().cpu().numpy()
                adv_egs.append((init_pred.item(), final_pred.item(), adv_eg))
        else:
            if len(adv_egs) < 5:
                adv_eg = perturbed_data.squeeze().detach().cpu().numpy()
                adv_egs.append((init_pred.item(), final_pred.item(), adv_eg))

    final_acc = correct / float(len(test_loader))
    print('Epsilon: %.4f Test Accuracy； %d / %d = %.3f' % (eps, correct, len(test_loader), final_acc))
    return final_acc, adv_egs


def aecurve(eps, accs):
    plt.figure(figsize=(5, 5))
    plt.plot(eps, accs, '*-')
    plt.xticks(np.arange(0, 0.012, step=0.001))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title('Accuracy & Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    PublicTestset = FER2013(split='PublicTest', transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=bs, shuffle=False, num_workers=1)
    PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=bs, shuffle=False, num_workers=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Device: ', device)
    model = ResNet18()
    checkpoint = torch.load(model_path)
    print('Loading model from %s' % model_path)
    model.load_state_dict(checkpoint['net'])
    # best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    # best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    # best_PublicTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    # best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']

    model.to(device)
    model.eval()
    accs = []
    egs = []
    for e in epls:
        acc, eg = test(model, device, PrivateTestloader, e)
        accs.append(acc)
        egs.append(eg)
    aecurve(eps=epls, accs=accs)





