import os
import torch
import pandas as pd
import numpy as np
from homework_model_modification import LinearRegressionModel, MultiClassLogisticRegression


def preprocess_insurance(df):
    numeric_cols = ['age', 'bmi', 'children']
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / (std + 1e-8)

    for col in ['sex', 'region']:
        unique = sorted(df[col].dropna().unique())
        mapping = {val: idx for idx, val in enumerate(unique)}
        df[col] = df[col].map(mapping)

    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    return df

def get_features_and_targets_regression(df):
    X = df.drop(columns=['charges']).values
    y = df['charges'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

def get_features_and_targets_classification(df):
    X = df.drop(columns=['smoker']).values
    y = df['smoker'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def load_linear_model(path, in_features):
    model = LinearRegressionModel(in_features)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def load_multiclass_model(path, in_features, num_classes):
    model = MultiClassLogisticRegression(in_features, num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def mean_squared_error_manual(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def precision_recall_f1_manual(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def test_linear_regression(model, X_test, y_test):
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
    mse = mean_squared_error_manual(y_test.numpy(), y_pred.numpy())
    print(f"Linear Regression Test MSE: {mse:.4f}")

def test_multiclass_logistic(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)  # shape (N, 2)
        preds = logits.argmax(dim=1).cpu().numpy()
    y_true = y_test.cpu().numpy()
    precision, recall, f1 = precision_recall_f1_manual(y_true, preds)
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'insurance1.csv')
    if data_path is None:
        print("Файл insurance1.csv не найден в папке data")
        return

    df_test = pd.read_csv(data_path)
    df_test = preprocess_insurance(df_test)

    # Регрессия
    X_reg, y_reg = get_features_and_targets_regression(df_test)
    in_features_reg = X_reg.shape[1]
    linear_model = load_linear_model(os.path.join(script_dir, 'models', 'best_linear_regression.pth'), in_features_reg)
    test_linear_regression(linear_model, X_reg, y_reg)

    # Классификация
    X_clf, y_clf = get_features_and_targets_classification(df_test)
    in_features_clf = X_clf.shape[1]
    num_classes = 2
    logistic_model = load_multiclass_model(os.path.join(script_dir, 'models', 'best_multiclass_logistic_regression.pth'), in_features_clf, num_classes)
    test_multiclass_logistic(logistic_model, X_clf, y_clf)

if __name__ == "__main__":
    main()
