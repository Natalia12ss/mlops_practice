import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def main():

    """
    data_creation.py
    Создаёт синтетические наборы данных для обучения и тестирования модели (бинарная классификация)

    1. Создаёт папки "train" и "test" для хранения данных
    2. Генерирует 1000 объектов с 5 признаками и 2 классами (0 и 1)
    3. Добавляет небольшой шум
    4. Разделяет выборку: 80% на обучение, 20% на тестирование
    5. Сохраняет данные в 4 CSV-файла:
        - train/X.csv, train/y.csv
        - test/X.csv, test/y.csv
    """

    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)

    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_classes=2,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )

    noise = np.random.normal(0, 0.1, X.shape)
    X_noisy = X + noise

    split_idx = int(0.8 * len(X_noisy))

    X_train, X_test = X_noisy[:split_idx], X_noisy[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    pd.DataFrame(X_train).to_csv("train/X.csv", index=False, header=False)
    pd.DataFrame(y_train).to_csv("train/y.csv", index=False, header=False)

    pd.DataFrame(X_test).to_csv("test/X.csv", index=False, header=False)
    pd.DataFrame(y_test).to_csv("test/y.csv", index=False, header=False)

    print("data complete")


if __name__ == "__main__":
    main()
