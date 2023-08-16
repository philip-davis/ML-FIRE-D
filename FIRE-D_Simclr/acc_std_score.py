from evaluate_embedding import evaluate_embedding
import numpy as np
from PIL import Image
import os

fileList = ["./result/0-epoch50_16.npz",
            "./result/1-epoch50_16.npz",
            "./result/2-epoch50_16.npz",
            "./result/3-epoch50_16.npz",
            "./result/4-epoch50_16.npz",
            "./result/5-epoch50_16.npz",
            "./result/6-epoch50_16.npz",
            "./result/7-epoch50_16.npz",
            "./result/8-epoch50_16.npz",
            "./result/9-epoch50_16.npz",
            ]

accList = []
for fileName in fileList:
    y = np.load('wildfire_label.npz', allow_pickle=True)['arr_0']
    emb = np.load(fileName, allow_pickle=True)['arr_0']
    acc_val, acc = evaluate_embedding(emb, y)
    print(fileName, acc_val, acc)
    accList.append(acc_val)

accNp = np.array(accList)
print(accNp)
accResult = np.mean(accNp)
std = np.std(accNp)
print("acc",accResult)
print("std",std)