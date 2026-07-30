"""
Model training utilities for JobClarity.
"""

from xgboost import XGBClassifier


def create_model(
    random_state: int = 42,
):
    """
    Create a baseline XGBoost classifier.
    """

    model = XGBClassifier(
        random_state=random_state,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss"
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train the model.
    """

    model.fit(X_train, y_train)

    return model