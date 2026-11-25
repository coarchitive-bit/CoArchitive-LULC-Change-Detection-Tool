"""
Model catalogue for the platform: the user picks the algorithm at the
Classify step. Every model is trained on the same cleaned training table
(data/processed/training_features_v2.csv) and exposes predict_proba so it
plugs into the temporally consistent change detection.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR
from modules.features import FEATURE_NAMES

SEED = 42
TRAINING_TABLE = PROCESSED_DIR / "training_features_v2.csv"


class KMeansClassifier:
    """Unsupervised K-Means; clusters are named by their majority training label."""

    def __init__(self, k=12):
        self.scaler = StandardScaler()
        self.km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5)

    def fit(self, X, y):
        c = self.km.fit_predict(self.scaler.fit_transform(X))
        self.classes_ = np.unique(y)
        self.lookup = np.zeros(self.km.n_clusters, dtype=int)
        for k in range(self.km.n_clusters):
            vals, counts = np.unique(y[c == k], return_counts=True)
            self.lookup[k] = np.searchsorted(self.classes_, vals[counts.argmax()]) if len(vals) else 0
        return self

    def predict_proba(self, X):
        d = self.km.transform(self.scaler.transform(X))
        p = np.zeros((len(X), len(self.classes_)), dtype="float32")
        p[np.arange(len(X)), self.lookup[d.argmin(axis=1)]] = 1.0
        return p

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


MODELS = {
    "gbt": {
        "label": "Gradient-boosted trees",
        "short": "Recommended. Fast, compact, 87.6% accuracy (kappa 0.82).",
        "help": "HistGradientBoosting: hundreds of shallow decision trees, each correcting the "
                "previous ones. Best balance of accuracy, speed and model size for state-wide runs.",
        "factory": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED),
    },
    "rf": {
        "label": "Random Forest",
        "short": "Highest accuracy in our test (88.8%), slower on large areas.",
        "help": "An ensemble of deep decision trees voting together. The classic remote-sensing "
                "classifier; robust, but prediction is slower and the model is large.",
        "factory": lambda: RandomForestClassifier(
            n_estimators=120, max_depth=22, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    },
    "mlp": {
        "label": "Neural network (MLP)",
        "short": "Deep-learning baseline on pixels, 88.0% accuracy.",
        "help": "A two-layer neural network (128-64 neurons) on the 14 pixel features. "
                "The entry point for the deep-learning roadmap (U-Net / foundation models).",
        "factory": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300, random_state=SEED)),
    },
    "svm": {
        "label": "Support Vector Machine",
        "short": "Kernel SVM (RBF, approximated for speed), about 88% accuracy.",
        "help": "Support Vector Machine with an approximated RBF kernel (Nystroem features + "
                "linear SVM), so it can classify millions of pixels in seconds.",
        "factory": lambda: make_pipeline(StandardScaler(), Nystroem(n_components=400, random_state=SEED),
                                         SGDClassifier(loss="modified_huber", class_weight="balanced",
                                                       max_iter=50, random_state=SEED)),
    },
    "kmeans": {
        "label": "K-Means (unsupervised)",
        "short": "No training labels needed, 82.4% accuracy. Baseline only.",
        "help": "Groups pixels into 12 spectral clusters without labels, then names each cluster "
                "by its majority training class. Shows how much supervision adds.",
        "factory": lambda: KMeansClassifier(),
    },
}


def train(model_key, table=TRAINING_TABLE):
    df = pd.read_csv(Path(table))
    X, y = df[FEATURE_NAMES].values, df["class_id"].values
    return MODELS[model_key]["factory"]().fit(X, y)
