import numpy as np
from src.ml.dataset import save_dataset_hdf5
from src.ml.baseline import load_flattened_subset, train_and_evaluate, compare_single_vs_multimodal


def _make_synthetic_h5(tmp_path, n_per_class=20, energy_bins=6, time_bins=6, seed=0):
    rng = np.random.default_rng(seed)
    labels = ["standard"] * n_per_class + ["liv"] * n_per_class + ["alp"] * n_per_class
    n = len(labels)
    # give each class a distinct mean so a classifier can actually separate them
    offsets = {"standard": 0.0, "liv": 5.0, "alp": -5.0}
    tensors = np.stack([
        rng.normal(loc=offsets[label], scale=1.0, size=(3, energy_bins, time_bins))
        for label in labels
    ]).astype(np.float32)

    path = tmp_path / "synthetic.h5"
    save_dataset_hdf5(tensors, labels, path)
    return path


def test_load_flattened_subset_shapes(tmp_path):
    path = _make_synthetic_h5(tmp_path, n_per_class=10, energy_bins=4, time_bins=4)

    X_all, y_all, label_map = load_flattened_subset(path, n_per_class=10, modality="all")
    assert X_all.shape == (30, 3 * 4 * 4)
    assert y_all.shape == (30,)
    assert set(label_map.keys()) == {"standard", "liv", "alp"}

    X_spectral, y_spectral, _ = load_flattened_subset(path, n_per_class=10, modality="spectral")
    assert X_spectral.shape == (30, 4 * 4)


def test_load_flattened_subset_respects_n_per_class_cap(tmp_path):
    path = _make_synthetic_h5(tmp_path, n_per_class=10, energy_bins=3, time_bins=3)

    X, y, label_map = load_flattened_subset(path, n_per_class=4, modality="all")
    assert X.shape[0] == 12  # 4 per class * 3 classes
    counts = np.bincount(y)
    assert all(c == 4 for c in counts)


def test_train_and_evaluate_separates_well_separated_classes(tmp_path):
    path = _make_synthetic_h5(tmp_path, n_per_class=40, energy_bins=5, time_bins=5)
    X, y, label_map = load_flattened_subset(path, n_per_class=40, modality="all")

    result = train_and_evaluate(X, y, label_map, seed=0)
    assert result["confusion_matrix"].shape == (3, 3)
    assert result["roc_auc_macro"] > 0.9  # classes are trivially separable by construction


def test_compare_single_vs_multimodal_returns_both_settings(tmp_path):
    path = _make_synthetic_h5(tmp_path, n_per_class=30, energy_bins=4, time_bins=4)

    results = compare_single_vs_multimodal(path, n_per_class=30, seed=0)
    assert set(results.keys()) == {"spectral", "all"}
    assert results["spectral"]["n_features"] == 4 * 4
    assert results["all"]["n_features"] == 3 * 4 * 4
