
import pandas as pd


def load_data(train_path, test_path):
    """Load Titanic training and test datasets."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data


def engineer_features(data):
    """Create engineered features from raw Titanic data."""
    data = data.copy()

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    data["Title"] = (
        data["Name"]
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .str.strip()
    )

    data["Title"] = data["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    common_titles = ["Mr", "Miss", "Mrs", "Master", "Dr", "Rev"]

    data["Title"] = data["Title"].where(
        data["Title"].isin(common_titles), "Rare"
    )

    data["Deck"] = data["Cabin"].str[0].fillna("Unknown")

    data["TicketGroupSize"] = (
        data.groupby("Ticket")["Ticket"].transform("size")
    )

    return data


def get_feature_columns():
    """Return available feature combinations."""
    return {
        "Original 4 features": [
            "Pclass", "Sex", "SibSp", "Parch"
        ],
        "Add Age and Fare": [
            "Pclass", "Sex", "SibSp", "Parch", "Age", "Fare"
        ],
        "Add family features": [
            "Pclass", "Sex", "SibSp", "Parch",
            "Age", "Fare", "FamilySize", "IsAlone"
        ],
        "All engineered features": [
            "Pclass", "Sex", "Age", "SibSp", "Parch",
            "Fare", "Embarked", "FamilySize", "IsAlone",
            "Title", "Deck", "TicketGroupSize"
        ]
    }


def prepare_features(data, feature_set):
    """Prepare selected features and encode categorical variables."""
    data = engineer_features(data)

    feature_columns = get_feature_columns()

    X = data[feature_columns[feature_set]].copy()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    X = pd.get_dummies(
        X,
        columns=list(categorical_columns),
        dtype=int
    )

    return X
