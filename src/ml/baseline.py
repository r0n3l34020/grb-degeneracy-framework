from pathlib import Path

import h5py
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

# === Day 9 ===
# Baseline classifiers: flatten tensors and train a gradient-boosted-tree model
# (sklearn's HistGradientBoostingClassifier — same family as XGBoost; XGBoost itself
# can't load on this machine without installing Homebrew first, see conversation),
# comparing single-modality (spectral channel only) vs multi-modality (all 3 channels)
# input, which is the central question this whole framework is built to answer.


def load_flattened_subset(h5_path, n_per_class: int = 5000, modality: str = "all", seed: int = 0):
    """
    Stratified-sample up to `n_per_class` events per physics_model directly from a
    consolidated HDF5 file (only the sampled rows are read — the full file, e.g. the
    Day 8 suite, is ~9GB and is never loaded into memory at once).

    modality: "all" -> flatten all 3 channels (multi-modality); "spectral" -> channel 0 only.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        label_map = json.loads(f.attrs["label_map"])
        labels = f["labels"][:]

        selected = []
        for class_id in sorted(label_map.values()):
            class_indices = np.flatnonzero(labels == class_id)
            n_take = min(n_per_class, len(class_indices))
            selected.append(rng.choice(class_indices, size=n_take, replace=False))
        selected = np.sort(np.concatenate(selected))  # h5py fancy-indexing requires increasing order

        y = labels[selected]
        if modality == "spectral":
            X = f["tensors"][selected, 0, :, :]  # channel 0 = spectral-temporal flux map
        elif modality == "all":
            X = f["tensors"][selected]
        else:
            raise ValueError(f"modality must be 'all' or 'spectral', got {modality!r}")

    X = X.reshape(X.shape[0], -1).astype(np.float32)
    return X, y, label_map


def train_and_evaluate(X: np.ndarray, y: np.ndarray, label_map: dict, test_size: float = 0.25, seed: int = 0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    class_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc_macro": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
        "report": classification_report(y_test, y_pred, target_names=class_names, zero_division=0),
    }


def compare_single_vs_multimodal(h5_path, n_per_class: int = 5000, seed: int = 0):
    """Train and evaluate the same baseline model on spectral-only vs full multi-modal input."""
    results = {}
    for modality in ("spectral", "all"):
        X, y, label_map = load_flattened_subset(h5_path, n_per_class=n_per_class, modality=modality, seed=seed)
        results[modality] = train_and_evaluate(X, y, label_map, seed=seed)
        results[modality]["n_features"] = X.shape[1]
    return results


if __name__ == "__main__":
    import time

    h5_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "events_suite.h5"
    if not h5_path.exists():
        h5_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "events.h5"
    print(f"Using dataset: {h5_path}")

    start = time.perf_counter()
    results = compare_single_vs_multimodal(h5_path, n_per_class=5000)
    elapsed = time.perf_counter() - start
    print(f"Trained + evaluated both models in {elapsed:.1f}s")

    for modality, result in results.items():
        print(f"\n=== {modality} ({result['n_features']} features) ===")
        print("Confusion matrix (rows=true, cols=pred):")
        print(result["confusion_matrix"])
        print(f"ROC-AUC (macro, one-vs-rest): {result['roc_auc_macro']:.4f}")
        print(result["report"])

    from src.utils.visualizer import plot_confusion_matrix, plot_roc_curves
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for col, modality in enumerate(("spectral", "all")):
        result = results[modality]
        plot_confusion_matrix(result["confusion_matrix"], result["class_names"], ax=axes[0, col],
                               title=f"{modality} ({result['n_features']} features)")
        plot_roc_curves(result["y_test"], result["y_proba"], result["class_names"], ax=axes[1, col],
                         title=f"{modality} ROC (AUC={result['roc_auc_macro']:.3f})")
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "day9_baseline_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison figure to {out_dir / 'day9_baseline_comparison.png'}")
