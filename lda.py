# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv("Wine.csv")

x = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#PCA
from sklearn.decomposition import PCA
pca =PCA(n_components = 2)
X_train2 = pca.fit_transform(X_train)
#Train ile aynı boyut ve koordinatlar için sadece transform
X_test2 = pca.transform(X_test)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
#Kolonların etkisine göre 2 boyuta çekiyor.O nedenle y_train var
X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

#İlk data(13boyutlu)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#After PCA
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#After LDA
classifier3 = LogisticRegression(random_state=0)
classifier3.fit(X_train_lda,y_train)

#Tahminler
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)
y_pred_lda = classifier3.predict(X_test_lda)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("13 boyutlu ilk data")
print(cm)

cm2 = confusion_matrix(y_test,y_pred2)
print("With PCA")
print(cm2)

cm3 = confusion_matrix(y_pred,y_pred2)
print(" Comparing two Results ")
print(cm3)

cm4 = confusion_matrix(y_test,y_pred_lda)
print(" With LDA ")
print(cm4)























