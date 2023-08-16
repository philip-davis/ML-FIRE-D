#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 18:15:14 2023

@author: zhiwei
"""


from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import ToTensor
import rasterio
import os
import torch
import os 
import numpy as np
import torch
from evaluate_embedding import evaluate_embedding



# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

sum(p.numel() for p in model.parameters())


# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms

labels = np.loadtxt('labels_NASA.txt')
labels =torch.tensor( labels)

file_list1 = os.listdir('labeled/uniform/')
labels =np.array(labels.detach().cpu().numpy())
res = [0]*10
for j in range(1):
    torch.manual_seed(j)
    for i in range(3400):
        image = rasterio.open('labeled/uniform/'+file_list1[i])
        image = image.read()
        image = ToTensor()(image)
        image = image.reshape(1,366,366)
        zeros = torch.zeros(2,366,366)
        image = torch.cat((image,zeros),0)
        batch = preprocess(image).unsqueeze(0)
        # Step 4: Use the model and print the predicted category
        prediction = model(batch)
        torch.save(prediction,'embedding_resnet_50/'+str(i)+'.pt')
    file_list = os.listdir('embedding_resnet_50')
    ans= torch.load('embedding_resnet_50/'+file_list[0])
    for i in range(1,3400):
        file = torch.load('embedding_resnet_50/'+file_list[i])
        ans=torch.cat((ans,file),0)
    
    emb = np.array(ans.detach().cpu().numpy())
    res[j] = evaluate_embedding(emb,labels)[0]
np.mean(res)
np.std(res)
