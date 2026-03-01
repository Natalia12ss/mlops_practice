import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


def main():

    """
    model_testing.py 

    Загружаем обученную модель и тестовые данные, 
    выполняем предсказание и вычисляем accuracy
    """

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv("test/X_scaled.csv", header=None).values
    y_test = pd.read_csv("test/y_scaled.csv", header=None).values.ravel()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Точности модели: {acc:.3f}")


if __name__ == "__main__":
    main()
