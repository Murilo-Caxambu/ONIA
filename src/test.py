import os
import sys

# Adicionar o diretório pai de 'src' ao sys.path para permitir importações absolutas
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# Importar os módulos usando importações absolutas
from dados import load_data
from model_training import train_model
from predict import make_predictions

def main():
    # Carregar os dados
    df_treino, df_teste = load_data()

    # Treinar o modelo
    clf = train_model(df_treino)

    # Fazer previsões e salvar em um arquivo CSV
    make_predictions(clf, df_teste)

if __name__ == "__main__":
    main()