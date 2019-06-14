from fer import FER2013
from torch.autograd import Variable
from models import *
import torch
import torch.nn as nn
import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

apply_defend = True  # to disable defend function, set this to be False
apply_attack = True  # to diable attack function, set this to be False
cut_size = 44
bs = 1
models = ['VGG13/PrivateTest_model2.t7', 'Resnet18/PrivateTest_model2.t7']
model_path = models[1]

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


def fgsm_defense(img, eps, device):
    # noise = np.random.normal(size=img.shape)
    noise = torch.randn(img.shape, dtype=torch.float32, device=device)
    defended_img = img + eps * noise
    defended_img = torch.clamp(defended_img, 0, 1)
    return defended_img


def fgsm_attack(img, eps, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_img = img + eps * sign_data_grad
    perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img


def test(model, device, test_loader, eps):
    correct = 0
    adv_egs = []
    test_loss = 0
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
        test_loss += loss.item()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, eps, data_grad)
        if apply_defend:
            perturbed_data = fgsm_defense(perturbed_data, 0.02, device)

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
    print(test_loss / len(test_loader))
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
    # device = torch.device('cpu')
    print('Device: ', device)
    if model_path == models[1]:
        model = ResNet18()
    else:
        model = VGG('VGG13')
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
    print(epls, accs)

    cnt = 0
    plt.figure(figsize=(8, 11))
    for i in range(len(epls)):
        for j in range(len(egs[i])):
            cnt += 1
            plt.subplot(len(epls), len(egs[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if i == len(epls) - 1:
                plt.ylabel("Eps: %.3f" % epls[i], fontsize=12)
            elif j == 0:
                    plt.ylabel(" %.3f" % epls[i], fontsize=12)
            orig, adv, ex = egs[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex.transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()
    print('Finish')
