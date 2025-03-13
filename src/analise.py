# src/analise.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar dados
df_treino = pd.read_csv('../data/treino.csv')
df_teste = pd.read_csv('../data/teste.csv')

# Verificar as primeiras linhas
print("Dados de Treino:")
print(df_treino.head())

print("\nDados de Teste:")
print(df_teste.head())

# Verificar informações básicas
print("\nInformações dos Dados de Treino:")
print(df_treino.info())

print("\nInformações dos Dados de Teste:")
print(df_teste.info())

# Verificar valores faltantes
print("\nValores Faltantes nos Dados de Treino:")
print(df_treino.isnull().sum())

print("\nValores Faltantes nos Dados de Teste:")
print(df_teste.isnull().sum())

# Verificar distribuição das classes
print("\nDistribuição das Classes no Conjunto de Treino:")
print(df_treino['target'].value_counts())

# Plotar matriz de correlação
correlacao = df_treino.drop('id', axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.show()