# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:00:10 2020

@author: SeasonTaiInOTA
"""
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

print('-----------data loading------------')
df = pd.read_csv('covid_19.csv',index_col = [0])
df = df.drop(df.index[0:2])
df = df.drop(columns=['Lat', 'Long'])
df = df.astype(int)

for i in range(len(df)):
    for j in range(1,len(df.columns)):
        df.iloc[i][j-1]= df.iloc[i][j] - df.iloc[i][j-1]

df  = df.drop(columns= df.columns[-1])

print('-----------data loaded------------')

data = df.transpose()

corrMatrix = pd.DataFrame.corr(data)
#print (corrMatrix)
#plt.figure(figsize=(10,10))
#sn.heatmap(corrMatrix.iloc[:10,:10], annot=True)
#plt.show()

print('-----------data selecting------------')
threshold = 0.8
A = []

for i in range(len(corrMatrix)):
    cand = []
    for j in range(len(corrMatrix)):
        if corrMatrix.iloc[i][j] > threshold:
            cand.append(corrMatrix.columns.values[j])
    A.append(cand)
#
C = []
for i in range(len(corrMatrix)):
    if (len(A[i]) >20 ):
        C.append(i)

l = 80
new_data = np.zeros((len(C), l))
for i in range(len(C)):       
    index = C[i]
    for j in range(l):
        if df.iloc[index][j+1] > df.iloc[index][j]:
            new_data[i][j] = 1
        else:
            new_data[i][j] = 0
print('-----------data selected------------')


    
