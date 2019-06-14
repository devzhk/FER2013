import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from fer import FER2013
import transforms as transforms
from sklearn.metrics import confusion_matrix
from models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

models = ['VGG13/PrivateTest_model2.t7', 'Resnet18/PrivateTest_model2.t7']
model_path = models[1]
cut_size = 44
split = 'PublicTest'  # 'PrivateTest'

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


if __name__ == '__main__':
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    if model_path == models[0]:
        net = VGG('VGG13')
    elif model_path == models[1]:
        net = ResNet18()
    checkpoint = torch.load(model_path)

    net.load_state_dict(checkpoint['net'])
    net.to(device)
    net.eval()
    Testset = FER2013(split=split, transform=transform_test)
    Testloader = torch.utils.data.DataLoader(Testset, batch_size=32, shuffle=False, num_workers=1)
    correct = 0
    total = 0
    all_target = []
    for batch_idx, (inputs, targets) in enumerate(Testloader):

        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        _, predicted = torch.max(outputs_avg.data, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        if batch_idx == 0:
            all_predicted = predicted
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_targets = torch.cat((all_targets, targets), 0)

    acc = 100. * correct / total
    print("accuracy: %0.3f" % acc)

    # Compute confusion matrix
    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                          title=split+' Confusion Matrix (Accuracy: %0.3f%%)' %acc)
    plt.savefig('%s_public.png' % model_path)
    plt.close()
