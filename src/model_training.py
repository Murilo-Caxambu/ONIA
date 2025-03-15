import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(df_treino):
    # Separar as características (features) e o alvo (target)
    X = df_treino.drop(['id', 'target'], axis=1)
    y = df_treino['target']

    # Dividir os dados em conjuntos de treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo de Árvore de Decisão
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Fazer previsões no conjunto de validação
    y_pred = clf.predict(X_val)

    # Avaliar o modelo
    print("\nRelatório de Classificação:")
    print(classification_report(y_val, y_pred))

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_val, y_pred))

    # Plotar matriz de correlação
    correlacao = df_treino.drop('id', axis=1).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacao, annot=True, cmap='coolwarm')
    plt.title("Matriz de Correlação")
    plt.show()

    return clf