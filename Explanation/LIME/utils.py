from torch.nn import functional as F
import torch
from torch.autograd import Variable
#import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import os
from torchvision import models, transforms, utils

def load_model(model_name):

    if model_name =="vgg19":

        model = models.vgg19(pretrained=True)
        model.eval()
        if cuda_available():
            model.cuda()
    elif model_name=="resnet52":
        pass
    #for p in model.features.parameters():
    #    p.requires_grad = False
    #for p in model.classifier.parameters():
    #    p.requires_grad = False

    return model
def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def load_image(path):
   # img = cv2.imread(path, 1)
   # img = cv2.resize(img, (224, 224))
   # img = np.float32(img) / 255

    return img

def image_preprocessing(img):
    #img should be PIL image

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if cuda_available():
        img=preprocess(img).cuda()
    else:
        img=preprocess(img)

    img.unsqueeze_(0)

    return img
def to_tensor(var):
    if cuda_available():
        var=torch.tensor(var,dtype=torch.float32).cuda()
    else:
        var=torch.tensor(var,dtype=torch.float32)

    return var
def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if cuda_available():
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)