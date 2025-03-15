import pandas as pd
import os

def make_predictions(clf, df_teste):
    # Fazer previsões no conjunto de teste
    X_test = df_teste.drop('id', axis=1)
    y_test_pred = clf.predict(X_test)

    # Criar um DataFrame com as previsões
    df_pred = pd.DataFrame({'id': df_teste['id'], 'target': y_test_pred})

    # Salvar as previsões em um arquivo CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath(os.path.join(current_dir, '../data/predicoes.csv'))
    df_pred.to_csv(output_path, index=False)
    print(f"Previsões salvas em: {output_path}")