# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:25:03 2020

@author: Benjamim
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


'''
dados = pd.read_csv('C:/Users/Benjamim/Desktop/classificador/INFLUD-30-11-2020.csv',sep= ';')


dados = pd.read_csv('https://raw.githubusercontent.com/ChernoBen/IAatividadeII/main/new_dataset',
                 sep= ';')'''
df = dados[['NU_IDADE_N','FATOR_RISC','EVOLUCAO']]
df.head()
#removendo valores Nan
df = df.dropna()
dicio = ['EVOLUCAO']
def rotula(dataset,param):
    arr =[]
    param = param
    dt = dataset
    for item in dt:
        if item == param:
            arr.append(1)
        else:
            arr.append(0)
    return arr
for indice in range(len(dicio)):
    #para cada valor na posição INDICE remova valores diferentes de positivo e negativo para os casos
    df  = df.drop(df[df[dicio[indice]] > 2 ].index)
    df  = df.drop(df[df[dicio[indice]] < 1  ].index)
    df[dicio[indice]] = rotula(df[dicio[indice]],1)

#removendo valores Nan
df = df.dropna()
#rotulando lados 1 == cura
df['FATOR_RISC'] = rotula(df['FATOR_RISC'],'S')

X = df[['NU_IDADE_N','FATOR_RISC']].values
y = df['EVOLUCAO'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

X_test

# instancia o classificador com nome logit
logit = LogisticRegression()

# treina o modelo
logit.fit(X_train, y_train)

# faz predicao e salva em y_pred
logit.predict(X_test)
y_pred = logit.predict(X_test)

# acuracia
acuracia = logit.score(X_test, y_test)

# matriz de confusao
print(confusion_matrix(y_test, y_pred))
matriz_de_confusao = confusion_matrix(y_test, y_pred)

# computa probabilidades
y_pred_prob = logit.predict_proba(X_test)[:,1]

# Gera fpr, tpr e thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# curva ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
sns.set_theme(style="darkgrid")






