import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

path = 'Dataset'

X = []
Y = []
'''
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img,(150,150),interpolation = cv2.INTER_CUBIC)
            image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit = 5)
            img = clahe.apply(image_bw) + 30
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            X.append(img.ravel())
            if name == 'InfectedFish':
                Y.append(1)
            if name == 'FreshFish':
                Y.append(0)

X = np.asarray(X)
Y = np.asarray(Y)
np.save("model/X",X)
np.save("model/Y",Y)
'''
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")

X = X.astype('float32')
X = X/255
    
test = X[3].reshape(150,150,3)
cv2.imshow("aa",cv2.resize(test,(200,200)))
cv2.waitKey(0)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

print(X.shape)
print(Y.shape)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

svm_cls = SVC(kernel='linear')
svm_cls.fit(X, Y)
y_pred = svm_cls.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



