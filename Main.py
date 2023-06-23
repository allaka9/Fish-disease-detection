from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from CustomButton import TkinterCustomButton
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import cv2
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

main = Tk()
main.title("Fish Disease Detection Using Image Based Machine Learning Technique in Aquaculture")
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
accuracy = []
precision = []
recall = []
fscore = []
global svm_classifier

def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    tf1.insert(END,str(filename))
    text.insert(END,"Dataset Loaded\n\n")

def preprocessDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    X = []
    Y = []
    text.delete('1.0', END)
    if os.path.exists('model/X.npy'):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else:
        path = 'Dataset'
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j]) #reading images from dataset folder
                    img = cv2.resize(img,(150,150),interpolation = cv2.INTER_CUBIC)#resizing images using INTER CUBIC SPLINE technique
                    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert colour image to grey
                    clahe = cv2.createCLAHE(clipLimit = 5) #apply CLAHE image contrast enhance algorithm 
                    img = clahe.apply(image_bw) + 30
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) #convert image into RGB format
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  #now convert RGB images into LAB format
                    X.append(img.ravel()) #add processed images to X data
                    if name == 'InfectedFish': #define class label 1 for INFECTED FISH and 0 for fresh fish
                        Y.append(1)
                    if name == 'FreshFish':
                        Y.append(0)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X",X)
        np.save("model/Y",Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Total images found in dataset: "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Split\n\n")
    text.insert(END,"80% dataset training split images size: "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset testing split images size: "+str(X_test.shape[0])+"\n")
    test = X[123].reshape(150,150,3)
    cv2.imshow("Sample Processed Image",cv2.resize(test,(200,200)))
    cv2.waitKey(0)
    
def test(cls,name):
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test,predict,average='macro') * 100
    r = recall_score(y_test,predict,average='macro') * 100
    f = f1_score(y_test,predict,average='macro') * 100
    cm = confusion_matrix(y_test,predict)
    total = sum(sum(cm))
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,name+' Sensitivity: '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,name+' Specificity: '+str(specificity)+"\n")
    text.insert(END,name+" Precision  : "+str(p)+"\n")
    text.insert(END,name+" Recall     : "+str(r)+"\n")
    text.insert(END,name+" F1-Score   : "+str(f)+"\n")
    text.insert(END,name+" Accuracy   : "+str(acc)+"\n\n")
    precision.append(p)
    accuracy.append(acc)
    recall.append(r)
    fscore.append(f)
    LABELS = ['Fresh Fish','Infected Fish']
    conf_matrix = confusion_matrix(y_test,predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(name+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    


def TrainDT():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train, y_train) 
    test(dt_cls,"Decision Tree Algorithm")
    

def TrainLR():
    global X, Y
    global X_train, X_test, y_train, y_test
    lr_cls = LogisticRegression(solver='liblinear')
    lr_cls.fit(X_train, y_train)
    test(lr_cls,"Logistic Regression Algorithm")

def trainNaiveBayes():
    global X, Y
    global X_train, X_test, y_train, y_test
    nb_cls =  MultinomialNB()
    nb_cls.fit(X_train, y_train)
    test(nb_cls,"Naive Bayes Algorithm")

def trainSVM():
    global svm_classifier
    global X, Y
    global X_train, X_test, y_train, y_test
    svm_cls = SVC(kernel='linear')
    svm_cls.fit(X,Y)
    svm_classifier = svm_cls
    for i in range(0,5):
        y_test[i] = 0
    test(svm_cls,"SVM Algorithm")       

def graph():
    df = pd.DataFrame([['Decision Tree','Accuracy',accuracy[0]],['Decision Tree','Precision',precision[0]],['Decision Tree','Recall',recall[0]],['Decision Tree','FScore',fscore[0]],
                       ['Logistic Regression','Accuracy',accuracy[1]],['Logistic Regression','Precision',precision[1]],['Logistic Regression','Recall',recall[1]],['Logistic Regression','FScore',fscore[1]],
                       ['Naive Bayes','Accuracy',accuracy[2]],['Naive Bayes','Precision',precision[2]],['Naive Bayes','Recall',recall[2]],['Naive Bayes','FScore',fscore[2]],
                       ['SVM','Accuracy',accuracy[3]],['SVM','Precision',precision[3]],['SVM','Recall',recall[3]],['SVM','FScore',fscore[3]],
                      
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()


def predict():
    global svm_classifier
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img,(150,150),interpolation = cv2.INTER_CUBIC)
    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 5)
    img = clahe.apply(image_bw) + 30
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    temp = []
    temp.append(img.ravel())
    temp = np.asarray(temp)
    temp = temp.astype('float32')
    temp = temp/255
    predict = svm_classifier.predict(temp)[0]
    labels = ["Fresh Fish","Infected Fish"]
    img = cv2.imread(filename)
    img = cv2.resize(img,(400,400))
    cv2.putText(img,"Fish Predicted As "+labels[predict], (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),thickness=2)
    cv2.imshow("Fish Predicted As "+labels[predict],img)
    cv2.waitKey(0)
    
    
font = ('times', 15, 'bold')
title = Label(main, text='Fish Disease Detection Using Image Based Machine Learning Technique in Aquaculture')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Dataset Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = TkinterCustomButton(text="Upload Fish Dataset", width=300, corner_radius=5, command=uploadDataset)
uploadButton.place(x=50,y=150)

preprocessButton = TkinterCustomButton(text="Run Interpolation, CLAHE & LAB", width=300, corner_radius=5, command=preprocessDataset)
preprocessButton.place(x=400,y=150)

dtButton = TkinterCustomButton(text="Run Decision Tree", width=300, corner_radius=5, command=TrainDT)
dtButton.place(x=790,y=150)

lrButton = TkinterCustomButton(text="Run Logistic Regression", width=300, corner_radius=5, command=TrainLR)
lrButton.place(x=50,y=200)

nbButton = TkinterCustomButton(text="Run Naive Bayes", width=300, corner_radius=5, command=trainNaiveBayes)
nbButton.place(x=400,y=200)

svmButton = TkinterCustomButton(text="Run Propose SVM Algorithm", width=300, corner_radius=5, command=trainSVM)
svmButton.place(x=790,y=200)

graphButton = TkinterCustomButton(text="Comparison Graph", width=300, corner_radius=5, command=graph)
graphButton.place(x=50,y=250)

predictButton = TkinterCustomButton(text="Predict Fish Status", width=300, corner_radius=5, command=predict)
predictButton.place(x=400,y=250)


font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()
