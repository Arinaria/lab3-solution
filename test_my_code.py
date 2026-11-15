import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models

df_reg = pd.read_csv('Student_Performance.csv')
df_reg['Extracurricular Activities'] = df_reg['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

X_reg = df_reg.drop('Performance Index', axis=1)
y_reg = df_reg['Performance Index']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

reg_input = RegressionInput(
    train_features=X_train_r_scaled, train_target=y_train_r,
    test_features=X_test_r_scaled, test_target=y_test_r,
    learning_rate=0.05, epochs=800
)
reg_result = train_linear_regression(reg_input)

print(f"Final MSE: {reg_result.loss_history[-1]:.4f}")
print(f"Predictions: {reg_result.predictions[:5]}")

df_cls = pd.read_csv('german.csv', sep=';')
y_cls = df_cls.iloc[:, 0]
X_cls = df_cls.iloc[:, 1:]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

cls_input = ClassificationInput(
    train_features=X_train_c_scaled, train_target=y_train_c,
    test_features=X_test_c_scaled, test_target=y_test_c,
    learning_rate=0.1, epochs=1000
)
cls_result = train_logistic_classifier(cls_input)
print(f"Final Cross-Entropy Loss: {cls_result.loss_history[-1]:.4f}")

comparison_input = ClassificationInput(
    train_features=X_train_c_scaled, train_target=y_train_c,
    test_features=X_test_c_scaled, test_target=y_test_c
)
comparison_result = compare_classical_models(comparison_input)

print(f"LogReg F1: {comparison_result.logistic_regression.f1:.4f}")
print(f"Decision Tree F1: {comparison_result.decision_tree.f1:.4f}")
print(f"KNN F1: {comparison_result.knn.f1:.4f}")