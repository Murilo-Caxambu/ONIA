import pandas as pd
import numpy as np
import os

def create_sample_data():
    # Define as features
    features = ['TempMédia', 'Gravidade', 'PressãoAtm', 'Radiação', 'ComposiçãoAr',
                'Hidratação', 'Vegetação', 'Fauna', 'SoloFértil', 'Ventos',
                'Luas', 'Magnetismo', 'ClimaEstável']
    
    # Cria dados de treino
    np.random.seed(42)
    n_treino = 10501
    df_treino = pd.DataFrame({
        'id': range(1, n_treino + 1),
        **{f: np.random.rand(n_treino) * 10 for f in features},
        'target': np.random.randint(0, 5, n_treino)
    })

    # Cria dados de teste
    n_teste = 4501
    df_teste = pd.DataFrame({
        'id': range(1, n_teste + 1),
        **{f: np.random.rand(n_teste) * 10 for f in features},
    })

    # Salva os arquivos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '../data')
    
    # Cria o diretório data se não existir
    os.makedirs(data_dir, exist_ok=True)
    
    # Salva os arquivos
    df_treino.to_csv(os.path.join(data_dir, 'treino.csv'), index=False)
    df_teste.to_csv(os.path.join(data_dir, 'teste.csv'), index=False)
    
    print("Arquivos criados com sucesso!")
    print(f"treino.csv: {len(df_treino)} linhas")
    print(f"teste.csv: {len(df_teste)} linhas")

if __name__ == "__main__":
    create_sample_data()