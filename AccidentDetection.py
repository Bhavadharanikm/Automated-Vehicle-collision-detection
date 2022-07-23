import webbrowser

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
from PIL import Image, ImageFile
from torch import nn
from torch import optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms

import cv2

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')

else:
    print('CUDA is available!  Training on GPU ...')
ImageFile.LOAD_TRUNCATED_IMAGES = True
#!pip install --upgrade wandb




test_transforms = transforms.Compose([transforms.Resize(255),
                                      #  transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      ])

model = models.densenet161()


model.classifier = nn.Sequential(nn.Linear(2208, 1000),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1000, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# model = model.cuda()
model.load_state_dict(torch.load('tensorboardexp.pt', 'cpu'))
classes = ["accident", "noaccident"]
# model.load_state_dict(torch.load('tensorboardexp.pt'))
count = 0
counts = 1

def accidentDetection(frame):
    try:
        img = Image.fromarray(frame)
    except ValueError:
        return
    except AttributeError:
        return
    img = test_transforms(img)
    img = img.unsqueeze(dim=0)
    # img = img.cuda()
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

        index = int(predicted.item())

        return 'status: ' + classes[index]
