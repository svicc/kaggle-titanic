import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from utils import load_data, prepare_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------


TRAIN_PATH = "datasets/train.csv"
TEST_PATH = "datasets/test.csv"

FEATURE_SETS = [
    "Original 4 features",
    "Add Age and Fare",
    "Add family features",
    "All engineered features"
]


# --------------------------------------------------
# Model
# --------------------------------------------------

def build_random_forest(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=1
):
    """Build a Random Forest classification pipeline."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=1,
            n_jobs=-1
        ))
    ])


# --------------------------------------------------
# Holdout validation
# --------------------------------------------------

def evaluate_holdout(model, X_train, X_cv, y_train, y_cv):
    """Train and evaluate a model on a holdout validation set."""

    model.fit(X_train, y_train)

    predictions = model.predict(X_cv)
    probabilities = model.predict_proba(X_cv)[:, 1]

    return {
        "Accuracy": accuracy_score(y_cv, predictions),
        "Precision": precision_score(
            y_cv, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_cv, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_cv, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_cv, probabilities)
    }


def compare_feature_sets_holdout(train_data):
    """Compare feature sets using one identical train/validation split."""

    y = train_data["Survived"]

    train_idx, cv_idx = train_test_split(
        train_data.index,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = []

    for name in FEATURE_SETS:
        X = prepare_features(train_data, name)

        X_train = X.loc[train_idx]
        X_cv = X.loc[cv_idx]

        y_train = y.loc[train_idx]
        y_cv = y.loc[cv_idx]

        # Match validation columns to training columns.
        X_cv = X_cv.reindex(
            columns=X_train.columns,
            fill_value=0
        )

        model = build_random_forest()

        metrics = evaluate_holdout(
            model, X_train, X_cv, y_train, y_cv
        )

        results.append({
            "Feature set": name,
            "Num features": X_train.shape[1],
            **metrics
        })

    results_df = pd.DataFrame(results)

    return results_df.sort_values(
        "Accuracy", ascending=False
    )


# --------------------------------------------------
# Cross-validation
# --------------------------------------------------

def evaluate_cross_validation(model, X, y, cv, scoring):
    """Evaluate a model using cross-validation."""

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results = {}

    for metric in scoring:
        values = scores[f"test_{metric}"]

        results[f"{metric.title()} Mean"] = values.mean()
        results[f"{metric.title()} Std"] = values.std()

    return results


def compare_feature_sets_cv(train_data):
    """Compare feature sets using stratified five-fold CV."""

    y = train_data["Survived"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = []

    for name in FEATURE_SETS:
        X = prepare_features(train_data, name)

        model = build_random_forest()

        metrics = evaluate_cross_validation(
            model, X, y, cv, scoring
        )

        results.append({
            "Feature set": name,
            "Num features": X.shape[1],
            **metrics
        })

    results_df = pd.DataFrame(results)

    return results_df.sort_values(
        "Accuracy Mean", ascending=False
    )


# --------------------------------------------------
# Hyperparameter tuning
# --------------------------------------------------

def tune_random_forest(train_data):
    """Find the best Random Forest hyperparameters using GridSearchCV."""

    X = prepare_features(
        train_data, "All engineered features"
    )
    y = train_data["Survived"]

    pipeline = build_random_forest(
        n_estimators=300
    )

    param_grid = {
        "rf__max_depth": [4, 5, 7, 10, None],
        "rf__min_samples_leaf": [1, 2, 4]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X, y)

    return grid


def display_tuning_results(grid):
    """Display the best parameters and top tuning results."""

    print("Best parameters:", grid.best_params_)
    print("Best CV accuracy:", round(grid.best_score_, 4))

    results = pd.DataFrame(grid.cv_results_)

    columns = [
        "param_rf__max_depth",
        "param_rf__min_samples_leaf",
        "mean_train_score",
        "mean_test_score",
        "std_test_score"
    ]

    print(
        results[columns]
        .sort_values("mean_test_score", ascending=False)
        .head(10)
        .round(4)
        .to_string(index=False)
    )


# --------------------------------------------------
# Final model and Kaggle submission
# --------------------------------------------------

def train_final_model(train_data):
    """Train the final model using all labeled training data."""

    X = prepare_features(
        train_data, "All engineered features"
    )
    y = train_data["Survived"]

    model = build_random_forest(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=4
    )

    model.fit(X, y)

    return model, X.columns


def create_submission(model, train_columns, test_data):
    """Generate and save the Kaggle submission file."""

    X_test = prepare_features(
        test_data, "All engineered features"
    )

    X_test = X_test.reindex(
        columns=train_columns,
        fill_value=0
    )

    predictions = model.predict(X_test)

    output = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

    output.to_csv("submission.csv", index=False)

    print(output.head())
    print("Submission shape:", output.shape)
    print("Saved submission.csv")

    return output


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    train_data, test_data = load_data(
        TRAIN_PATH, TEST_PATH
    )

    # 1. Holdout feature comparison
    holdout_results = compare_feature_sets_holdout(train_data)
    print("\nHoldout Results:")
    print(holdout_results.round(4).to_string(index=False))

    # 2. Five-fold cross-validation
    cv_results = compare_feature_sets_cv(train_data)
    print("\nCross-Validation Results:")
    print(cv_results.round(4).to_string(index=False))

    # 3. Hyperparameter tuning
    grid = tune_random_forest(train_data)
    display_tuning_results(grid)

    # 4. Final model and submission
    model, train_columns = train_final_model(train_data)
    create_submission(model, train_columns, test_data)


if __name__ == "__main__":
    main()
