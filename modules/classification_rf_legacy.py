
import numpy as np
import pandas as pd
import joblib

from config import BASE_DIR, OUTPUT_DIR

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "lulc_training_samples.csv"
)

MODEL_DIR = BASE_DIR / "models"
STATS_DIR = OUTPUT_DIR / "statistics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = (
    MODEL_DIR / "lulc_random_forest_6class_5bands.joblib"
)

FEATURES = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "NDVI", "NDWI", "NDBI", "BSI"
]

CLASS_NAMES = {
    1: "Built-up",
    2: "Tree cover",
    3: "Agriculture",
    4: "Water",
    5: "Scrub",
    6: "Barren"
}

RANDOM_SEED = 42


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

def load_training_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    data = pd.read_csv(INPUT_FILE)

    required = FEATURES + [
        "class_id",
        "source_year",
        "polygon_id"
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.dropna(subset=required).copy()

    data["class_id"] = data["class_id"].astype(int)
    data["source_year"] = data["source_year"].astype(int)

    data = data[
        data["class_id"].isin(CLASS_NAMES.keys()) &
        data["source_year"].isin([2020, 2025])
    ].copy()

    if data["class_id"].nunique() != 6:
        raise ValueError("All six classes must be present.")

    if not np.isfinite(
        data[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid spectral values detected.")

    # Keep polygons from different years separate
    data["group_id"] = (
        data["source_year"].astype(str)
        + "_"
        + data["polygon_id"].astype(str)
    )

    print("\nSix-class dataset loaded.")
    print("Total pixels:", len(data))

    print("\nClass distribution:")
    print(data.groupby(
        ["source_year", "class_name"]
    ).size())

    return data


# -------------------------------------------------
# POLYGON-LEVEL VALIDATION SPLIT
# -------------------------------------------------

def split_data(data):

    X = data[FEATURES]
    y = data["class_id"]
    groups = data["group_id"]

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    train_idx, validation_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx]
    X_validation = X.iloc[validation_idx]

    y_train = y.iloc[train_idx]
    y_validation = y.iloc[validation_idx]

    train_groups = set(groups.iloc[train_idx])
    validation_groups = set(groups.iloc[validation_idx])

    if train_groups.intersection(validation_groups):
        raise ValueError("Polygon leakage detected.")

    expected_classes = set(CLASS_NAMES.keys())

    if set(y_train.unique()) != expected_classes:
        raise ValueError("Training split is missing a class.")

    if set(y_validation.unique()) != expected_classes:
        raise ValueError(
            "Validation split is missing a class. "
            "We may need to adjust the fold strategy."
        )

    print("\nPolygon-level split completed.")
    print("Training pixels:", len(X_train))
    print("Validation pixels:", len(X_validation))
    print("Training polygons:", len(train_groups))
    print("Validation polygons:", len(validation_groups))

    print("\nValidation class support:")
    print(y_validation.map(CLASS_NAMES).value_counts())

    return X_train, X_validation, y_train, y_validation


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

def train_model(X_train, y_train):

    print("\nTraining six-class Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Training completed.")

    return model


# -------------------------------------------------
# EVALUATE MODEL
# -------------------------------------------------

def evaluate_model(model, X_validation, y_validation):

    predictions = model.predict(X_validation)

    labels = list(CLASS_NAMES.keys())

    accuracy = accuracy_score(
        y_validation, predictions
    )

    balanced = balanced_accuracy_score(
        y_validation, predictions
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_validation,
        predictions,
        labels=labels,
        target_names=[
            CLASS_NAMES[x] for x in labels
        ],
        zero_division=0
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=labels
    )

    print("\n--------------------------------")
    print("SIX-CLASS MODEL RESULTS")
    print("--------------------------------")

    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification report:")
    print(report)

    matrix_df = pd.DataFrame(
        matrix,
        index=[CLASS_NAMES[x] for x in labels],
        columns=[CLASS_NAMES[x] for x in labels]
    )

    matrix_df.to_csv(
        STATS_DIR / "confusion_matrix_6class.csv"
    )

    with open(
        STATS_DIR / "classification_report_6class.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(f"Overall accuracy: {accuracy:.4f}\n")
        file.write(f"Balanced accuracy: {balanced:.4f}\n")
        file.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
        file.write(report)

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance.to_csv(
        STATS_DIR / "feature_importance_6class.csv",
        index=False
    )

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    return accuracy, balanced, macro_f1


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    data = load_training_data()

    X_train, X_validation, y_train, y_validation = (
        split_data(data)
    )

    model = train_model(X_train, y_train)

    evaluate_model(
        model,
        X_validation,
        y_validation
    )

    joblib.dump(model, MODEL_FILE)

    print("\nSix-class model saved:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()
