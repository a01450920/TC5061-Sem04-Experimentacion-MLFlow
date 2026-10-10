## TC5061 - MLOps | Código de experimentación base con MLFlow

## Elaborado por

A01450920 - Roberto Iván Ruiz Cantú

## Descripción

Proyecto de experimentación y seguimiento sobre un modelo de Machine Learning, utilizando MLflow y Scikit-Learn.

Se emplea un árbol de decisión (Decision Tree Classifier) para predecir la supervivencia de pasajeros del conjunto de datos Titanic.


## Ejecución

### Consultar ayuda

El script permite configurar distintos hiperparámetros mediante argumentos de línea de comandos.

Para consultar las opciones disponibles:

```bash
python src/train.py --help
```

### Parámetros disponibles

| Parámetro | Tipo | Valor por defecto | Descripción |
|------------|------|-------------------|-------------|
| `--max-depth` | int | 4 | Profundidad máxima del árbol de decisión |
| `--min-samples-split` | int | 2 | Número mínimo de muestras requerido para dividir un nodo |
| `--criterion` | string | gini | Criterio de división (`gini` o `entropy`) |
| `--seed` | int | 42 | Semilla para reproducibilidad |

### Corrida base

```bash
python src/train.py --seed 42
```

### Ejemplos de experimentos

```bash
python src/train.py --seed 42 --max-depth 10
python src/train.py --seed 42 --max-depth 10 --criterion entropy
python src/train.py --seed 42 --max-depth 10 --min-samples-split 5

```
