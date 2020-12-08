import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline


dados = pd.read_csv('https://raw.githubusercontent.com/ChernoBen/IAatividadeII/main/new_dataset',
                 sep= ';')
df = dados[['NU_IDADE_N','FATOR_RISC','EVOLUCAO']]
df.head()

#remoçãode casos com valores diferentes de obito e cura
df  = df.drop(df[df['EVOLUCAO'] > 2 ].index)
df  = df.drop(df[df['EVOLUCAO'] < 1  ].index)
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
#rotulando lados
df['FATOR_RISC'] = rotula(df['FATOR_RISC'],'S')
df['EVOLUCAO'] = rotula(df['EVOLUCAO'],1)
#separa bases em instacias de obito e cura
positivo = df[df['EVOLUCAO'].isin([1])]
negativo = df[df['EVOLUCAO'].isin([0])]
positivo = df[df['FATOR_RISC'].isin([1])]
negativo = df[df['FATOR_RISC'].isin([0])]

#verificação de disperçao de dados
fig,ax = plt.subplots(figsize=(6,6))
ax.scatter(positivo['NU_IDADE_N'],positivo['FATOR_RISC'],s=50,c='b',marker='o',label='Cura')
ax.scatter(negativo['NU_IDADE_N'],negativo['FATOR_RISC'],s=50,c='r',marker='x',label='Obito')
ax.legend()
ax.set_xlabel('curas')
ax.set_ylabel('obitos')

#pré processamento de dados
n_features = len(df.columns)-1

X = np.array(df.drop('EVOLUCAO',1))
y = df.iloc[:,n_features:n_features+1].values

#declarando veotresd de media e desvio padrão na padronização para classificação futura

mean = X.mean(axis=0)
std = X.std(axis=1)

#padronização dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#criando X-zero com valores =1
def insert_ones(X):
	ones = np.ones([X.shape[0],1])
	return np.concatenate((ones,X),axis=1)

#criando vetor de W,com valores randomicos entre 0 e 1 baseado na quantidade de features
W = np.random.rand(1,n_features)

#craindo sigmoid

def sigmoid(z):
	return 1/(1+np.exp(-z))

#visualizando sigmoid com distribuição de dados aleatorios
nums = np.arange(-10,10,step=1)
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(nums,sigmoid(nums),'r')

#função de custo aplicado a regressão logistica
def binary_cross_entropy(W,X,y):
	m = len(X)
	#ativado quando Y = 1
	part1 = np.multiply(-y,np.log(sigmoid(X @ W.T)))
	#ativado quando  Y = 0
	part2 = np.multiply((1- y),np.log(1-sigmoid(X @ W.T)))

	somatorio = np.sum(part1-part2)

	return somatorio/m

#implementação do gradiente
def gradiente_descendente(W,X,y,alpha,epoch):
	cost = np.zeros(epoch)
	for i in range(epoch):
        #rodapé 
		W = W - (alpha/len(X)) * np.sum(sigmoid(X @ W.T- y)*X,axis=1)
		#custo
		custo[i] = binary_cross_entropy(W,X,y)
		return W,custo



#inicializando treinamento
X = insert_ones(X)
alpha = 0.01 #taxa de aprendizado
epoch = 10000 #quantidade de repetições
W,custo = gradiente_descendente(W,X,y,alpha,epoch)


#visualizando custo

fig,ax = plt.subplots()
ax.plot(np.arange(epoch),custo,'r')
ax.set_xlabel('Iterações')
ax.set_ylabel('custo')
ax.set_tittle('Erro vs Epoch')

W
#executando predições
#retorna 1 se o retorno da sigmoid foi acima de 0.5
def predi(W,X,limear=0.5):
	p = sigmoid(W @ W.T) >= limear
	return (p.astype('int'))

#classificação de um novo paciente
#paciente de 45 anos e 1 para fator de risco
paciente1 = np.array([[45,1]])
#realizando padronização pela media e desvio padrão
paciente1 = (paciente1-mean)/std
paciente1 = insert_ones(paciente1)

#verificando a probabilidade de morrer ou não
sigmoid(paciente1 @ W.T)

#verificando a predição
predi(W,paciente1)

#rodapé
'''
1*
You are transposing a Matrix with 3 rows and 1 column to a Matrix with 3 columns and 1 row. 
Then you are multplying it with a similar Matrix (also 3 columns 1 row) which is incorrect mathematically. 
So you can either remove the transpose function or define your R Matrix as 1 row 3 columns and then transpose it. 
Check this for further information.
'''











