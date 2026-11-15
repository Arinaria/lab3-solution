from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from grader_contracts.linear_models import (
    RegressionInput, RegressionResult,
    ClassificationInput, ClassificationResult,
    ClassificationMetrics, BaselineComparisonResult
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    m, n = X_train.shape
    weights = np.zeros(n)
    bias = 0.0
    loss_history = []

    lr = data.learning_rate
    epochs = data.epochs

    for _ in range(epochs):
        y_pred = X_train @ weights + bias
        error = y_pred - y_train

        dw = (1 / m) * (X_train.T @ error)
        db = (1 / m) * np.sum(error)

        weights -= lr * dw
        bias -= lr * db

        mse = np.mean(error ** 2)
        loss_history.append(float(mse))

    test_predictions = X_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_predictions
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    m, n = X_train.shape
    weights = np.zeros(n)
    bias = 0.0
    loss_history = []

    lr = data.learning_rate
    epochs = data.epochs
    epsilon = 1e-15

    for _ in range(epochs):
        z = X_train @ weights + bias
        y_pred_proba = _sigmoid(z)

        error = y_pred_proba - y_train
        dw = (1 / m) * (X_train.T @ error)
        db = (1 / m) * np.sum(error)

        weights -= lr * dw
        bias -= lr * db

        loss = -np.mean(y_train * np.log(y_pred_proba + epsilon) +
                        (1 - y_train) * np.log(1 - y_pred_proba + epsilon))
        loss_history.append(float(loss))

    test_z = X_test @ weights + bias
    test_probabilities = _sigmoid(test_z)
    test_predictions = (test_probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions
    )


def _calculate_metrics(y_true, y_pred, y_proba):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0))
    )


def compare_classical_models(data) -> BaselineComparisonResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=float).ravel()

    lr_input = ClassificationInput(
        train_features=X_train, train_target=y_train,
        test_features=X_test, test_target=y_test
    )
    lr_result = train_logistic_classifier(lr_input)
    lr_metrics = _calculate_metrics(y_test, lr_result.predictions, lr_result.probabilities)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_metrics = _calculate_metrics(y_test, dt.predict(X_test), dt.predict_proba(X_test)[:, 1])

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_metrics = _calculate_metrics(y_test, knn.predict(X_test), knn.predict_proba(X_test)[:, 1])

    return BaselineComparisonResult(
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )