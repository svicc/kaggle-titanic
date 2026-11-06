import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from display import save_results
from error_analysis import (
    gender_pclass_error_analysis,
)
from utils import load_data, prepare_features

DEFAULT_TRAIN_PATH = "datasets/train.csv"
DEFAULT_TEST_PATH = "datasets/test.csv"
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5


# ============================================================
# Model definitions
# ============================================================


def build_models(random_state=1):
    """Create fresh model pipelines for the three classifiers."""
    return {
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        max_features=0.8,
                        min_samples_leaf=4,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        l2_regularization=0,
                        learning_rate=0.03,
                        max_iter=200,
                        max_leaf_nodes=15,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# Cross-validation
# ============================================================


def evaluate_fold(
    train_fold,
    valid_fold,
    models,
    fold_number,
):
    """Fit/evaluate every model on one fold and return result records."""
    y_train = train_fold["Survived"]
    y_valid = valid_fold["Survived"]

    # Calculate ticket counts from this fold's training partition only.
    ticket_counts = train_fold.groupby("Ticket").size()

    X_train = prepare_features(train_fold, "All engineered features")
    X_valid = prepare_features(valid_fold, "All engineered features")
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

    fold_results = []

    for model_name, base_model in models.items():
        model = clone(base_model)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        valid_predictions = model.predict(X_valid)
        valid_probabilities = model.predict_proba(X_valid)[:, 1]

        fold_results.append(
            {
                "Model": model_name,
                "Fold": fold_number,
                "PassengerId": valid_fold["PassengerId"].to_numpy(),
                "Actual": y_valid.to_numpy(),
                "Predicted": valid_predictions,
                "Survival_Probability": valid_probabilities,
                "Correct": (y_valid.to_numpy() == valid_predictions),
                "Train Accuracy": accuracy_score(y_train, train_predictions),
                "Validation Accuracy": accuracy_score(y_valid, valid_predictions),
                "ROC-AUC": roc_auc_score(y_valid, valid_probabilities),
            }
        )

    return fold_results


def run_cross_validation(
    train_data,
    models=None,
    n_splits=DEFAULT_N_SPLITS,
    random_state=DEFAULT_RANDOM_STATE,
    verbose=True,
):
    """
    Run stratified cross-validation for all models.

    Returns a DataFrame containing one row per model/fold. The prediction
    columns contain arrays, one value per passenger in that validation fold.
    """
    if models is None:
        models = build_models()

    y = train_data["Survived"]

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_fold_results = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        cv.split(train_data, y),
        start=1,
    ):
        if verbose:
            print(f"Processing fold {fold_number}/{n_splits}...")

        train_fold = train_data.iloc[train_idx].copy()
        valid_fold = train_data.iloc[valid_idx].copy()

        fold_results = evaluate_fold(
            train_fold=train_fold,
            valid_fold=valid_fold,
            models=models,
            fold_number=fold_number,
        )
        all_fold_results.extend(fold_results)

    return pd.DataFrame(all_fold_results)


# ============================================================
#  Results and summary
# ============================================================


def summarize_results(fold_df):
    """Calculate mean and standard deviation of each model's CV metrics."""
    summary = (
        fold_df.groupby("Model")
        .agg(
            Mean_Train_Accuracy=("Train Accuracy", "mean"),
            Mean_CV_Accuracy=("Validation Accuracy", "mean"),
            CV_Std=("Validation Accuracy", lambda x: x.std(ddof=0)),
            Mean_ROC_AUC=("ROC-AUC", "mean"),
            ROC_AUC_Std=("ROC-AUC", lambda x: x.std(ddof=0)),
        )
        .sort_values("Mean_CV_Accuracy", ascending=False)
    )
    return summary


# ============================================================
# Prediction analysis
# ============================================================


def expand_model_predictions(fold_df, model_name):
    """
    Convert one model's fold-level array columns into passenger-level rows.

    The returned DataFrame has one row per passenger across all validation
    folds, with the prediction and probability for that passenger.
    """
    model_results = fold_df[fold_df["Model"] == model_name]

    if model_results.empty:
        raise ValueError(f"No results found for model: {model_name}")

    records = []

    for _, row in model_results.iterrows():
        fold_predictions = pd.DataFrame(
            {
                "PassengerId": row["PassengerId"],
                "Actual": row["Actual"],
                "Predicted": row["Predicted"],
                "Survival_Probability": row["Survival_Probability"],
                "Correct": row["Correct"],
            }
        )
        fold_predictions["Fold"] = row["Fold"]
        records.append(fold_predictions)

    return pd.concat(records, ignore_index=True)


def add_passenger_details(predictions, train_data):
    """Attach all original passenger information to predictions."""
    return predictions.merge(
        train_data,
        on="PassengerId",
        how="left",
    )


def get_predictions_and_errors(fold_df, train_data):

    rf_predictions = expand_model_predictions(
        fold_df,
        model_name="Random Forest",
    )
    rf_predictions = add_passenger_details(
        rf_predictions,
        train_data,
    )

    rf_errors = rf_predictions.loc[~rf_predictions["Correct"]].copy()
    return rf_predictions, rf_errors


def main():
    train_data, test_data = load_data(DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH)

    models = build_models()

    fold_df = run_cross_validation(
        train_data=train_data,
        models=models,
    )

    summary = summarize_results(fold_df)
    # print_results(fold_df, summary)
    save_results(fold_df, summary)

    rf_predictions, rf_errors = get_predictions_and_errors(fold_df, train_data)
    gender_pclass_error_analysis(rf_predictions, rf_errors)


if __name__ == "__main__":
    main()
