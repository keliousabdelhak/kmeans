# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:39:52 2020

@author: k.abdelhak
"""

from sklearn.cluster import KMeans 
import numpy as np
from matplotlib import pyplot as plt


#read my data 
my_liste=[]
with open("weight_height.csv") as file:
	for i in file :
		n=i.split(",")
		my_liste.append([float(n[1]),float(n[2])])
x2=np.array(my_liste)


#predict my data
n=2 
kmeans=KMeans(n_clusters=n).fit(x2)
k=kmeans.predict(x2)
print(k)

#plot my data
plt.scatter(x2[:,0],x2[:,1],c=k)
    

