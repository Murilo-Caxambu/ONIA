import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Carrega os dados de treinamento e teste.
    O arquivo de treino deve ter 10.501 linhas e o arquivo de teste 4.501 linhas.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    treino_path = os.path.abspath(os.path.join(current_dir, '../data/treino.csv'))
    teste_path = os.path.abspath(os.path.join(current_dir, '../data/teste.csv'))

    print(f"Tentando carregar arquivo de treino: {treino_path}")
    print(f"Tentando carregar arquivo de teste: {teste_path}")

    # Verifica se os arquivos existem
    if not os.path.exists(treino_path):
        raise FileNotFoundError(f"Arquivo de treino não encontrado em: {treino_path}")
    if not os.path.exists(teste_path):
        raise FileNotFoundError(f"Arquivo de teste não encontrado em: {teste_path}")

    # Carrega os dados
    df_treino = pd.read_csv(treino_path)
    df_teste = pd.read_csv(teste_path)

    print(f"\nInformações do arquivo de treino:")
    print(f"Número de linhas: {len(df_treino)}")
    print(f"Número de colunas: {len(df_treino.columns)}")
    print(f"Colunas encontradas: {df_treino.columns.tolist()}")

    print(f"\nInformações do arquivo de teste:")
    print(f"Número de linhas: {len(df_teste)}")
    print(f"Número de colunas: {len(df_teste.columns)}")
    print(f"Colunas encontradas: {df_teste.columns.tolist()}")

    # Verifica o número correto de linhas
    if len(df_treino) != 10501:
        raise ValueError(f"Arquivo de treino deve ter 10501 linhas, mas tem {len(df_treino)}")
    if len(df_teste) != 4501:
        raise ValueError(f"Arquivo de teste deve ter 4501 linhas, mas tem {len(df_teste)}")

    return df_treino, df_teste

def train_model(df_treino):
    """
    Treina o modelo usando Árvore de Decisão.
    """
    # Separa features e target usando os nomes corretos das colunas
    features = ['TempMédia', 'Gravidade', 'PressãoAtm', 'Radiação', 'ComposiçãoAr',
                'Hidratação', 'Vegetação', 'Fauna', 'SoloFértil', 'Ventos',
                'Luas', 'Magnetismo', 'ClimaEstável']
    
    X = df_treino[features]
    y = df_treino['target']

    # Verifica a distribuição dos targets
    print("\nDistribuição dos targets no conjunto de treino:")
    print(df_treino['target'].value_counts().sort_index())

    # Treina o modelo
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    return clf

def make_predictions(clf, df_teste):
    try:
        features = ['TempMédia', 'Gravidade', 'PressãoAtm', 'Radiação', 'ComposiçãoAr',
                   'Hidratação', 'Vegetação', 'Fauna', 'SoloFértil', 'Ventos',
                   'Luas', 'Magnetismo', 'ClimaEstável']
        
        X_test = df_teste[features]
        y_pred = clf.predict(X_test)

        # Gera os IDs corretos começando de 4508
        novos_ids = range(4508, 4508 + len(y_pred))

        # Cria DataFrame com os novos IDs
        output_df = pd.DataFrame({
            'id': novos_ids,
            'target': y_pred.astype(int)
        })

        # Define o caminho do arquivo
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'final17.csv')
        
        # Salva ordenado por ID
        output_df.to_csv(desktop_path, index=False, sep=',')
        
        print(f"\nArquivo salvo em: {desktop_path}")
        
        # Verificação detalhada do arquivo
        if os.path.exists(desktop_path):
            df_verificacao = pd.read_csv(desktop_path)
            print("\n=== Verificação do Arquivo ===")
            print(f"Número total de linhas: {len(df_verificacao)}")
            print(f"Primeiro ID: {df_verificacao['id'].min()}")
            print(f"Último ID: {df_verificacao['id'].max()}")
            print(f"\nDistribuição dos targets:")
            print(df_verificacao['target'].value_counts().sort_index())
            print("\nPrimeiros 5 registros:")
            print(df_verificacao.head())
            
            # Abre o explorador
            os.system(f'explorer /select,"{desktop_path}"')

    except Exception as e:
        print(f"Erro ao salvar o arquivo: {str(e)}")

def main():
    try:
        print("1. Carregando os dados...")
        df_treino, df_teste = load_data()

        print("\n2. Treinando o modelo...")
        clf = train_model(df_treino)

        print("\n3. Fazendo previsões e salvando resultados...")
        make_predictions(clf, df_teste)
        
        print("\nProcesso concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")

if __name__ == "__main__":
    main()