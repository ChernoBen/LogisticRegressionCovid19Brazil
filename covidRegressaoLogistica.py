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
from matplotlib import cm
import seaborn as sb
%matplotlib inline


#Carregando base de dados
'''base = pd.read_csv('C:/Users/Benjamim/Desktop/classificador/INFLUD-30-11-2020.csv',
                   sep=';')
dados = base[['NU_IDADE','FATOR_RISK','EVOLUCAO']]'''
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
dados = pd.read_csv('https://raw.githubusercontent.com/ChernoBen/IAatividadeII/main/new_dataset', sep= ';')
'''removendo valores diferentes de obitos e cura'''
dados  = dados.drop(dados[dados['EVOLUCAO'] > 2 ].index)
dados  = dados.drop(dados[dados['EVOLUCAO'] < 1  ].index)
#1-cura 2-obito / 0-cura 1-obito
dados['EVOLUCAO'] = rotula(dados['EVOLUCAO'],2)

plt.scatter(dados['NU_IDADE_N'],dados['EVOLUCAO'])
#base.describe()

#visualização do coeficiente de correlação entre o atributo "IDADE" e "evolucao"
#para saber se a relação é positiva e qual a força da correlação 
coef_view = np.corrcoef(dados.NU_IDADE_N,dados.EVOLUCAO)


#CRIACAO das variaveis X e Y (variavel independente e variavel dependente)
#transformação de X para o formato de matriz adicionando um novo eixo (newexis)

X = dados.iloc[:,1].values
#tranforma X para formato Matriz
X = X[:,np.newaxis]
y = dados.iloc[:,23].values
#X[:,14] = rotula(X[:,14],'S')
#dividindo bases dados em treino e teste

#Criação do modelo, treinamento e visualização dos coeficientes
modelo = LogisticRegression()
#treina
modelo.fit(X,y)
#visualiza inclinação e interceptação
modelo.coef_
modelo.intercept_

#gera grafico de disperção que mostra a linha de melhor ajuste para o modelo
plt.scatter(X,y)

#geração de novos dados para gerar a funcao sigmoide
#theta0
x_teste = np.linspace(-3,150,30000)
#implementacao da funcao sigmoide
def model(x):
    return 1/(1+ np.exp(-x))
#Geração de previsoes (variavel r) e visualização dos resultados / p criação do modelo
r = model(x_teste*modelo.coef_ + modelo.intercept_).ravel()
plt.plot(x_teste,r,color='red')

''''''
#carregamento da base de dados com os novos 
base_previsoes = x_teste
base_previsoes

#mudanda dos dados formato de matriz
obitos = base_previsoes
obitoss = obitos.reshape(-1,1)

#previsões e geração de nova base de dados com os valores originais e as previsoes
previsoes_teste = modelo.predict(obitos)
previsoes_teste

base_previsoes = np.column_stack((base_previsoes,previsoes_teste))
base_previsoes


'''sigmoide explica a correlação entre a idade e obito
não quer dizer que exista uma causalidade entre a idade e obitos mas sim que x idade

tasks:
    calcular erros
    utilizar train test split para a linha 84
    aplicar gradiente
    
 '''


