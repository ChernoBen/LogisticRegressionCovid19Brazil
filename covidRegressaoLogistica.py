# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:00:39 2020

@author: Benjamim
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#Carregando base de dados
'''base = pd.read_csv('C:/Users/Benjamim/Desktop/classificador/INFLUD-30-11-2020.csv',
                   sep=';')
dados = base[['NU_IDADE','FATOR_RISK','EVOLUCAO']]'''

dados = pd.read_csv('https://raw.githubusercontent.com/ChernoBen/IAatividadeII/main/new_dataset', sep= ';')
'''removendo valores diferentes de obitos e cura'''
dados  = dados.drop(dados[dados['EVOLUCAO'] > 2 ].index)
dados  = dados.drop(dados[dados['EVOLUCAO'] < 1  ].index)

plt.scatter(dados['NU_IDADE_N'],dados['EVOLUCAO'])
#base.describe()

#visualização do coeficiente de correlação entre o atributo "IDADE" e "evolucao"
#para saber se a relação é positiva e qual a força da correlação 
np.corrcoef(dados.NU_IDADE_N,dados.EVOLUCAO)


#CRIACAO das variaveis X e Y (variavel independente e variavel dependente)
#transformação de X para o formato de matriz adicionando um novo eixo (newexis)
def rotula(dataset):
    arr =[]
    dt = dataset
    for item in dt:
        if item == 'S':
            arr.append(1)
        else:
            arr.append(0)
    return arr

X = dados.iloc[:,1].values
#tranforma X para formato Matriz
X = X[:,np.newaxis]
y = dados.iloc[:,23].values
#X[:,14] = rotula(X[:,14])
#dividindo bases dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=42)

#Criação do modelo, treinamento e visualização dos coeficientes
modelo = LogisticRegression()
#treina
modelo.fit(X_train,y_train)
#visualiza inclinação e interceptação
modelo.coef_
modelo.intercept_

#gera grafico de disperção que mostra a linha de melhor ajuste para o modelo
plt.scatter(X,y)
#geração de novos dados para gerar a funcao sigmoide
x_teste = np.linspace(10,50,100)
#implementacao da funcao sigmoide
def model(x):
    return 1/(1+ np.exp(-x))
#Geração de previsoes (variavel r) e visualização dos resultados / p criação do modelo
r = model(X_test*modelo.coef_ + modelo.intercept_).ravel()
plt.plot(X_test,r,color='red')

#modelo criado agora fazer as previsoes

#carregamento da base de dados com os novos 
base_previsoes = X_test
base_previsoes

#mudanda dos dados formato de matriz
despesas = base_previsoes.iloc[:,1].values
despesas = despesas.reshape(-1,1)

#previsões e geração de nova base de dados com os valores originais e as previsoes
previsoes_teste = modelo.predict(despesas)
previsoes_teste

base_previsoes = np.column_stack((base_previsoes,previsoes_teste))
base_previsoes





