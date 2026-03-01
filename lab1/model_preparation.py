import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


def main():

    """
    model_preparation.py 
    Обучение модели LogisticRegression
    """

    X_train = pd.read_csv("train/X_scaled.csv", header=None).values
    y_train = pd.read_csv("train/y_scaled.csv", header=None).values.ravel()

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("model train complete")


if __name__ == "__main__":
    main()
