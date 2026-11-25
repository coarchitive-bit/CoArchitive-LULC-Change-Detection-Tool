"""
Algorithm benchmark - "efficacy of available AI/ML/DL algorithms".

All models see the same features and the same polygon-grouped 5-fold split
(no validation polygon is ever used for training), so the numbers are
directly comparable and honest.

Models
  K-Means (unsupervised baseline; clusters mapped to the majority class)
  Random Forest
  Gradient-boosted trees (HistGradientBoosting)   <- production model
  Support Vector Machine (RBF)
  Neural network (multi-layer perceptron, deep-learning baseline on pixels)

Output: outputs/statistics/algorithm_benchmark.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STATS_DIR  # noqa: E402
from modules.features import FEATURE_NAMES  # noqa: E402

SEED = 42
MAX_TRAIN_SVM = 6000


class KMeansMapped:
    """Unsupervised clustering; each cluster takes its majority training label."""

    def __init__(self, k=12):
        self.scaler = StandardScaler()
        self.km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5)

    def fit(self, X, y):
        Z = self.scaler.fit_transform(X)
        c = self.km.fit_predict(Z)
        self.map = {}
        for k in np.unique(c):
            vals, counts = np.unique(y[c == k], return_counts=True)
            self.map[k] = vals[counts.argmax()]
        return self

    def predict(self, X):
        c = self.km.predict(self.scaler.transform(X))
        return np.array([self.map.get(k, -1) for k in c])


MODELS = {
    "K-Means (unsupervised)": lambda: KMeansMapped(),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED),
    "Gradient-boosted trees": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED),
    "Support Vector Machine": lambda: make_pipeline(
        StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")),
    "Neural network (MLP)": lambda: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                        max_iter=300, random_state=SEED)),
}


def run_benchmark(df):
    X = df[FEATURE_NAMES].values
    y = df["class_id"].values
    groups = df["fid"].values
    rng = np.random.default_rng(SEED)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))

    rows = []
    for name, factory in MODELS.items():
        pred = np.zeros_like(y)
        t0 = time.time()
        for tr, te in splits:
            if name.startswith("Support") and len(tr) > MAX_TRAIN_SVM:
                tr = rng.choice(tr, MAX_TRAIN_SVM, replace=False)
            pred[te] = factory().fit(X[tr], y[tr]).predict(X[te])
        rows.append({
            "algorithm": name,
            "overall_accuracy": round(accuracy_score(y, pred), 4),
            "kappa": round(cohen_kappa_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "runtime_s": round(time.time() - t0, 1),
        })
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows).sort_values("kappa", ascending=False)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATS_DIR / "algorithm_benchmark.csv", index=False)
    return out


if __name__ == "__main__":
    from modules.classification import extract_samples, load_reflectance
    from modules.features import build_feature_stack
    from config import PERIODS

    refl = load_reflectance(list(PERIODS))
    stacks = {y: (build_feature_stack(a), g) for y, (a, g) in refl.items()}
    print(run_benchmark(extract_samples(stacks)).to_string(index=False))
