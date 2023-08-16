from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

import numpy as np

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


y = np.load('./wildfire_label.npz', allow_pickle=True)['arr_0']

num_clusters=2

result = np.array([[0,0],[0,0]])
for fileName in fileList:
    emb = np.load(fileName, allow_pickle=True)['arr_0']
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(emb)
    centers = kmeans.cluster_centers_
    y_kmeans = kmeans.predict(emb)
    temp = confusion_matrix(y,y_kmeans)
    print(temp)
    result=np.add(result,temp)

print("-----------------")
print(result)

confusion_matrix = result
FP = (confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))[0] 
FN = (confusion_matrix.sum(axis=1) - np.diag(confusion_matrix))[0]
TP = (np.diag(confusion_matrix))[0]
TN = confusion_matrix.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

print("=================")
print("FP,FN,TP,TN")
print(FP,FN,TP,TN)
print("TPR,TNR,PPV,NPV,FPR,FNR,FDR")
print(TPR,TNR,PPV,NPV,FPR,FNR,FDR)
print("F1")
print(2*TP / (2*TP+FP+FN))
acc = 0.9044117647058824
print(2*acc * TPR / (acc + TPR))
print(TPR)