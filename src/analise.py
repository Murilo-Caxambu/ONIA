import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Obter o caminho absoluto do diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construir o caminho absoluto para os arquivos CSV
treino_path = os.path.abspath(os.path.join(current_dir, '../data/treino.csv'))
teste_path = os.path.abspath(os.path.join(current_dir, '../data/teste.csv'))

# Imprimir os caminhos para verificação
print(f"Caminho do arquivo de treino: {treino_path}")
print(f"Caminho do arquivo de teste: {teste_path}")

# Verificar se o diretório 'data' existe e criar se não existir
data_dir = os.path.abspath(os.path.join(current_dir, '../data'))
if not os.path.exists(data_dir):
    print(f"Diretório 'data' não encontrado: {data_dir}")
    os.makedirs(data_dir)
    print(f"Diretório 'data' criado: {data_dir}")
else:
    print(f"Diretório 'data' encontrado: {data_dir}")

# Verificar se os arquivos existem
if not os.path.exists(treino_path):
    print(f"Arquivo de treino não encontrado: {treino_path}")
    # Criar um arquivo CSV de exemplo
    df_treino_exemplo = pd.DataFrame({'id': [1, 2], 'feature1': [10, 20], 'target': [0, 1]})
    df_treino_exemplo.to_csv(treino_path, index=False)
    print(f"Arquivo de treino de exemplo criado: {treino_path}")

if not os.path.exists(teste_path):
    print(f"Arquivo de teste não encontrado: {teste_path}")
    # Criar um arquivo CSV de exemplo
    df_teste_exemplo = pd.DataFrame({'id': [1, 2], 'feature1': [30, 40], 'target': [0, 1]})
    df_teste_exemplo.to_csv(teste_path, index=False)
    print(f"Arquivo de teste de exemplo criado: {teste_path}")

# Carregar os dados de treinamento
df_treino = pd.read_csv(treino_path, encoding='utf-8')

# Verificar as primeiras linhas dos dados de treinamento
print("Dados de Treino:")
print(df_treino.head())

# Verificar informações básicas dos dados de treinamento
print("\nInformações dos Dados de Treino:")
print(df_treino.info())

# Verificar valores faltantes nos dados de treinamento
print("\nValores Faltantes nos Dados de Treino:")
print(df_treino.isnull().sum())

# Verificar distribuição das classes no conjunto de treinamento
print("\nDistribuição das Classes no Conjunto de Treino:")
print(df_treino['target'].value_counts())

# Plotar matriz de correlação
correlacao = df_treino.drop('id', axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.show()