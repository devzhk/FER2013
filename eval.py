from fer import FER2013
from torch.autograd import Variable
from models import *
import torch
import torch.nn as nn
import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import utils

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

transform_eval = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
tbs = 32
trainset = FER2013(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=tbs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split='PublicTest', transform=transform_eval)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=tbs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split='PrivateTest', transform=transform_eval)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=tbs, shuffle=False, num_workers=1)
criterion = nn.CrossEntropyLoss()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0


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



def PublicTest(epoch, net):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)

        correct += (predicted == targets).sum().item()
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    PublicTest_acc = 100.*correct/total
    # writer.add_scalars('Loss', {'PublicTest': PublicTest_loss / len(PublicTestloader)}, epoch)
    # writer.add_scalars('Accuracy', {'PublicTest': PublicTest_acc}, epoch)
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch


def PrivateTest(epoch, net):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)

        correct += (predicted == targets).sum().item()
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total
    # writer.add_scalars('Loss', {'PrivateTest': PrivateTest_loss / len(PrivateTestloader)}, epoch)
    # writer.add_scalars('Accuracy', {'PrivateTest': PrivateTest_acc}, epoch)

    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch


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
    mode = 'test'

    PublicTestset1 = FER2013(split='PublicTest', transform=transform_test)
    PublicTestloader1 = torch.utils.data.DataLoader(PublicTestset1, batch_size=bs, shuffle=False, num_workers=1)
    PrivateTestset1 = FER2013(split='PrivateTest', transform=transform_test)
    PrivateTestloader1 = torch.utils.data.DataLoader(PrivateTestset1, batch_size=bs, shuffle=False, num_workers=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
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

    if mode == 'test':
        PublicTest(1, model)
        PrivateTest(1, model)
        exit()

    accs = []
    egs = []
    for e in epls:
        acc, eg = test(model, device, PrivateTestloader1, e)
        accs.append(acc)
        egs.append(eg)
    aecurve(eps=epls, accs=accs)
    print(epls, accs)

    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epls)):
        for j in range(len(egs[i])):
            cnt += 1
            plt.subplot(len(epls), len(egs[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epls[i]), fontsize=14)
            orig, adv, ex = egs[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()





