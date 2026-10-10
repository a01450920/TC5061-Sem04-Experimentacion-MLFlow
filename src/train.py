import argparse
import mlflow
import mlflow.sklearn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)

def load_data_from_csv(filepath):
    """
    Carga el dataset desde CSV
    """
    df = pd.read_csv(filepath)
    
    return df

def preprocess_data(df, seed):
    """
    Preprocesar los datos
    """
    #Eliminar duplicados
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_after = len(df)

    print(
        f"Duplicados eliminados: "
        f"{rows_before - rows_after}"
    )
    
    # Se elimina la variable Cabin, ya que tiene más de 75% de valores faltantes
    df.drop(columns=["Cabin"], inplace=True)
    
    # Definir variables
    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
    
    # Variable objetivo
    y = df["Survived"]
    
    # Variables predictoras
    X = df[features].copy()
    
    # Dividir los datos en los conjuntos de entrenamiento y pruebas
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed
    )
    
    # Imputación de valores faltantes usando solamente valores del conjunto train
    age_median = X_train["Age"].median()
    embarked_mode = X_train["Embarked"].mode()[0]
    
    X_train["Age"] = X_train["Age"].fillna(age_median)
    X_test["Age"] = X_test["Age"].fillna(age_median)

    X_train["Embarked"] = X_train["Embarked"].fillna(embarked_mode)
    X_test["Embarked"] = X_test["Embarked"].fillna(embarked_mode)
    
    # Convertir variables categóricas en numéricas mediante One-Hot Encoding
    X_train = pd.get_dummies(X_train, columns=["Sex", "Embarked"], dtype=int)
    X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked"], dtype=int)
    
    # Alinear columnas entre entrenamiento y prueba
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    
    return X_train, X_test, y_train, y_test

    
# Entrenar el modelo
def train_model(X_train, y_train, args):
    """
    Entrenar el modelo según los hiperparámetros indicados para la ejecución
    """
    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        criterion=args.criterion,
        random_state=args.seed)
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    
    plt.savefig("artifacts/confusion_matrix.png", bbox_inches="tight")
    
    plt.close()
    
    return {
        "accuracy": accuracy,
        "f1": f1
    }
    
def main():
    """
    Orquesta ejecución completa del experimento
    """
    # Preparar argumentos
    parser = argparse.ArgumentParser(
        description="Entrenamiento y evaluación de un arbol de decisión para Titanic, usando MLflow."
    )
    parser.add_argument("--max-depth", help="Profundidad máxima del árbol",type=int, default=4)
    parser.add_argument("--min-samples-split", help="Mínimo de muestras para dividir un nodo", type=int, default=2)
    parser.add_argument("--criterion", help="Criterio de división", choices=["gini", "entropy"], default="gini")
    parser.add_argument("--seed", help="Semilla para reproducibilidad", type=int, default=42)
    args = parser.parse_args()

    # Registrar experimento en MLflow
    mlflow.set_experiment("Titanic_DecisionTree_Experiment")

    with mlflow.start_run():
        # Leer datos
        df = load_data_from_csv("data/Titanic_train.csv")
        # Particionar los datos"
        X_train, X_test, y_train, y_test = preprocess_data(df, args.seed)
        
        # Entrenar modelo y obtener evaluación
        model = train_model(X_train, y_train, args)
        metrics = evaluate_model(model, X_test, y_test)
        
        # Registrar parámetros, métricas y artefactos en MLflow
        mlflow.log_params(vars(args))
        mlflow.log_param("features", ",".join(X_train.columns))
        mlflow.log_param("dataset", "Titanic_train.csv")
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_artifact("artifacts/confusion_matrix.png")
        mlflow.sklearn.log_model(model, name="TitanicModel", skops_trusted_types=["sklearn.tree._tree.Tree"])
        
        # Mostrar métricos de evaluación
        print("\nResultados:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
if __name__ == "__main__":
    main()