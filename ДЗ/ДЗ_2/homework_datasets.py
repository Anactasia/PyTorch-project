import pandas as pd
import torch
import numpy as np
from homework_model_modification import (
    train_linreg,
    train_logreg
)

class CustomDataset:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.process()

    def normalize(self, columns):
        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            self.df[col] = (self.df[col] - mean) / (std + 1e-8)

    def encode_column(self, column):
        unique = sorted(self.df[column].dropna().unique())
        mapping = {val: idx for idx, val in enumerate(unique)}
        self.df[column] = self.df[column].map(mapping)

    def process(self):
        # Нормализация числовых колонок
        numeric_cols = ['age', 'bmi', 'children']
        self.normalize(numeric_cols)

        # Кодирование категориальных колонок
        for col in ['sex', 'region']:
            self.encode_column(col)

        # Бинарное значение target для классификации
        self.df['smoker'] = self.df['smoker'].map({'yes': 1, 'no': 0})

    def get_features_and_targets_regression(self):
        X = self.df.drop(columns=['charges']).values
        y = self.df['charges'].values.reshape(-1, 1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_features_and_targets_classification(self):
        X = self.df.drop(columns=['smoker']).values
        y = self.df['smoker'].values
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)  # dtype long — для классов


if __name__ == '__main__':
    ds = CustomDataset('ДЗ/ДЗ_2/data/insurance.csv')

    X_reg, y_reg = ds.get_features_and_targets_regression()
    train_linreg(
        X_reg, y_reg, l1_lambda=0.001, l2_lambda=0.001, patience=5, epochs=100
    )


    X_clf, y_clf = ds.get_features_and_targets_classification()

# Проверка баланса классов
    unique, counts = torch.unique(y_clf, return_counts=True)
    print("Классовое распределение (smoker):", dict(zip(unique.tolist(), counts.tolist())))

    train_logreg(
        X_clf, y_clf, num_classes=2, epochs=50
    )
