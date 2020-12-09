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



dados = pd.read_csv('C:/Users/Benjamim/Desktop/bases/INFLUD-30-11-2020.csv',sep= ';')

'''
dados = pd.read_csv('https://raw.githubusercontent.com/ChernoBen/IAatividadeII/main/new_dataset',
                 sep= ';')'''


#Tratamento de dados da base original
df = dados[['NU_IDADE_N','FATOR_RISC','EVOLUCAO']]
df.head()

#removendo valores Nan (not a number)
df = df.dropna()
#Lista de indices para melhor manipulação de variaveis da base
dicio = ['EVOLUCAO']

#rolutador binario de intancias para matrizes, np.array's e etc. 
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
    # EM INDICE remova valores diferentes de positivo e negativo para os casos
    df  = df.drop(df[df[dicio[indice]] > 2 ].index)
    df  = df.drop(df[df[dicio[indice]] < 1  ].index)
    df[dicio[indice]] = rotula(df[dicio[indice]],1)
    

#rotulando lados 1 == cura
df['FATOR_RISC'] = rotula(df['FATOR_RISC'],'S')

#Divisão da base entre treino e teste
X = df[['NU_IDADE_N','FATOR_RISC']].values
y = df['EVOLUCAO'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)
X_test

#Regressão logistica
# instancia o classificador 
classificador = LogisticRegression()
# treina o modelo
classificador.fit(X_train, y_train)
# faz predicao e salva em y_pred
classificador.predict(X_test)
y_pred = classificador.predict(X_test)
# acuracia
acuracia = classificador.score(X_test, y_test)
# matriz de confusao
print(confusion_matrix(y_test, y_pred))
matriz_de_confusao = confusion_matrix(y_test, y_pred)

# computa probabilidades
y_pred_prob = classificador.predict_proba(X_test)[:,1]

novaBase = pd.DataFrame(X_test)
novaBase['Result'] = y_test

sns.pairplot(novaBase,hue='Result');
sns.pairplot(novaBase, hue = 'Result', diag_kind = 'kde',plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},size = 4);
plt.suptitle('Evolução de pacientes com covide Idade X Fator de risco', 
             size = 15);

sns.scatterplot(data=novaBase, x=0, y=1, hue='Result');


#sigmoide
def sigmoide(x):
    return 1/(1+np.exp(-x))

retaSigma = sigmoide(X_test * classificador.coef_ + classificador.intercept_).ravel()





