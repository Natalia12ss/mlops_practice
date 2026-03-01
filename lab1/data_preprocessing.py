import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


def main():

    """
    data_preprocessing.py
    предобработка данных

    выполняет нормализацию признаков с помощью StandardScaler;
    преобразует данные из папок train/ и test/ к единому масштабу;

    1. Загружаем сырые данные из CSV-файлов (X.csv, y.csv)
    2. Инициализируем StandardScaler для нормализации признаков
    3. Обучаем скалер на тренировочной выборке (fit_transform)
    4. Применяем скалер к тестовой выборке (transform) - без обучения
    5. Сохраняем нормализованные данные в X_scaled.csv и y_scaled.csv
    6. Сохраняем объект скалера в scaler.pkl для будущего использования
    """

    X_train = pd.read_csv("train/X.csv", header=None).values
    y_train = pd.read_csv("train/y.csv", header=None).values
    X_test = pd.read_csv("test/X.csv", header=None).values
    y_test = pd.read_csv("test/y.csv", header=None).values

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled).to_csv("train/X_scaled.csv", index=False, header=False)
    pd.DataFrame(X_test_scaled).to_csv("test/X_scaled.csv", index=False, header=False)

    pd.DataFrame(y_train).to_csv("train/y_scaled.csv", index=False, header=False)
    pd.DataFrame(y_test).to_csv("test/y_scaled.csv", index=False, header=False)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("data preproce complete")


if __name__ == "__main__":
    main()
