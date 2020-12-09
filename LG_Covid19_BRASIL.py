# -*- coding: utf-8 -*-
"""
Created on Tue Nov  25 21:25:03 2020

@author: Benjamim
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


dados = pd.read_csv('C:/Users/Benjamim/Desktop/bases/INFLUD-30-11-2020.csv',sep= ';')
#Tratamento de dados da base original
df = dados[['NU_IDADE_N','FATOR_RISC','UTI','EVOLUCAO']]
df.head()

#removendo valores Nan (not a number)
#a) Remoção de casos de SRAG não diagnosticados como COVID-19
df = df.dropna()
#Lista de indices para melhor manipulação de variaveis da base
#basicamente a maioria dos mortos tiveram que passar pela UTI
dicio = ['EVOLUCAO','UTI']
#rolutador binario de intancias para matrizes, np.array's e etc.
#b)Remoção de variáveis relacionadas ao óbito (data do óbito e número da declaração de óbito)
'''
1-cura
2-obito
3-obito outras causas
9-ignorado
''' 
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
    # o score muda dependendo da classe que for rolulada como 1
    df[dicio[indice]] = rotula(df[dicio[indice]],1)   
#rotulando lados 1 == cura
df['FATOR_RISC'] = rotula(df['FATOR_RISC'],'S')


#c) Seleção e tratamento de variáveis;
X = df[['NU_IDADE_N','FATOR_RISC','UTI']].values
y = df['EVOLUCAO'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

#d) Treino de classificador para classificar entre “óbito por COVID-19” e “cura” (informação da coluna “EVOLUCAO”);
#Regressão logistica
# instancia o classificador 
classificador = LogisticRegression()
# treina o classificador
classificador.fit(X_train, y_train)
# faz predicao e salva em previoes
classificador.predict(X_test)
previsoes = classificador.predict(X_test)

#e) Avaliar classificador e reportar a acurácia geral e por classe em conjunto de teste.
# acuracia
acuracia = classificador.score(X_test, y_test)
# matriz de confusao para resultados de teste e previsoes
print(confusion_matrix(y_test, previsoes))
matriz_de_confusao = confusion_matrix(y_test, previsoes)
# previsão de probabilidades
probabPrevi = classificador.predict_proba(X_test)[:,1]
print(acuracia,probabPrevi)

#grafico p/avaliação
novaBase = pd.DataFrame(X_test)
novaBase['Result'] = y_test
sns.pairplot(novaBase,hue='Result')







