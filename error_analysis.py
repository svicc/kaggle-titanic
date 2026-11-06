def get_most_confident_errors(predictions, n=None):
    """Return the n incorrect predictions with the highest confidence."""
    results = predictions.copy()

    results["Confidence"] = np.where(
        results["Survival_Probability"] >= 0.5,
        results["Survival_Probability"],
        1 - results["Survival_Probability"],
    )

    return (
        results.loc[~results["Correct"]]
        .sort_values("Confidence", ascending=False)
        .head(n)
    )


def save_most_confident_errors(
    predictions,
    output_path="most_confident_wrong_predictions.csv",
):
    """Save every incorrect prediction, sorted by confidence, to CSV."""
    errors = get_most_confident_errors(predictions, n=None)
    errors.to_csv(output_path, index=False)

    print(f"Saved {len(errors)} incorrect predictions to {output_path}")
    return errors


def get_model_errors(fold_df, model_name="Random Forest"):
    """Return all incorrectly classified passengers for one model."""
    predictions = expand_model_predictions(fold_df, model_name)
    return predictions.loc[~predictions["Correct"]].copy()


def gender_pclass_error_analysis(rf_predictions, rf_errors):
    error_analysis = rf_errors.groupby(
        ["Sex", "Pclass"]
    ).agg(
        Total=("PassengerId", "count"),
        ActualSurvivors=("Actual", "sum"),
        PredictedSurvivors=("Predicted", "sum"),
    ).reset_index()

    print(error_analysis)
    error_analysis = rf_predictions.groupby(
        ["Sex", "Pclass"]
    ).agg(
        Total=("PassengerId", "count"),
        ActualSurvivors=("Actual", "sum"),
        PredictedSurvivors=("Predicted", "sum"),
        ErrorRate=("Correct", lambda x: (~x).mean()),
    ).reset_index()

    print(error_analysis)


def get_passengers_wrong_in_all_models(fold_df, train_data):
    """Return passengers incorrectly predicted by every evaluated model."""
    prediction_records = []

    for _, row in fold_df.iterrows():
        result = pd.DataFrame({
            "PassengerId": row["PassengerId"],
            "Model": row["Model"],
            "Correct": row["Correct"],
        })
        prediction_records.append(result)

    prediction_df = pd.concat(prediction_records, ignore_index=True)

    prediction_table = prediction_df.pivot(
        index="PassengerId",
        columns="Model",
        values="Correct",
    )

    all_wrong_ids = prediction_table.index[
        prediction_table.eq(False).all(axis=1)
    ]

    passenger_details = train_data[
        train_data["PassengerId"].isin(all_wrong_ids)
    ].copy()

    return passenger_details, prediction_table


def split_false_positives_and_negatives(predictions):
    """
    Split incorrect predictions into:
      - false positives: predicted survived, actually died
      - false negatives: predicted died, actually survived
    """
    errors = predictions.loc[~predictions["Correct"]].copy()

    false_positives = errors[
        (errors["Actual"] == 0) & (errors["Predicted"] == 1)
    ].copy()

    false_negatives = errors[
        (errors["Actual"] == 1) & (errors["Predicted"] == 0)
    ].copy()

    return false_positives, false_negatives
