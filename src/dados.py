import pandas as pd
import os

def load_data():
    # Obter o caminho absoluto do diretório atual
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir o caminho absoluto para os arquivos CSV
    treino_path = os.path.abspath(os.path.join(current_dir, '../data/treino.csv'))
    teste_path = os.path.abspath(os.path.join(current_dir, '../data/teste.csv'))

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
        df_treino_exemplo = pd.DataFrame({'id': [1, 2], 'TempMédia': [10, 20], 'Gravidade': [1, 1], 'PressãoAtm': [1, 1], 'Radiação': [1, 1], 'ComposiçãoAr': [1, 1], 'Hidratação': [1, 1], 'Vegetação': [1, 1], 'Fauna': [1, 1], 'SoloFértil': [1, 1], 'Ventos': [1, 1], 'Luas': [1, 1], 'Magnetismo': [1, 1], 'ClimaEstável': [1, 1], 'target': [0, 1]})
        df_treino_exemplo.to_csv(treino_path, index=False)
        print(f"Arquivo de treino de exemplo criado: {treino_path}")

    if not os.path.exists(teste_path):
        print(f"Arquivo de teste não encontrado: {teste_path}")
        # Criar um arquivo CSV de exemplo
        df_teste_exemplo = pd.DataFrame({'id': [1, 2], 'TempMédia': [30, 40], 'Gravidade': [1, 1], 'PressãoAtm': [1, 1], 'Radiação': [1, 1], 'ComposiçãoAr': [1, 1], 'Hidratação': [1, 1], 'Vegetação': [1, 1], 'Fauna': [1, 1], 'SoloFértil': [1, 1], 'Ventos': [1, 1], 'Luas': [1, 1], 'Magnetismo': [1, 1], 'ClimaEstável': [1, 1]})
        df_teste_exemplo.to_csv(teste_path, index=False)
        print(f"Arquivo de teste de exemplo criado: {teste_path}")

    # Carregar os dados de treinamento e teste
    df_treino = pd.read_csv(treino_path, encoding='utf-8')
    df_teste = pd.read_csv(teste_path, encoding='utf-8')

    return df_treino, df_teste