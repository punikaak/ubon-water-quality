"""Turbidity inference pipeline: MLP (calibrated), the best-performing MLP variant
trained on Ubon PCD stations, per user request.

Verified pipeline: raw Sentinel-2 bands/indices -> scaler_ubon_operational.pkl ->
best_model_neural_network_mlp.pkl -> calibrator_neural_network_mlp.pkl reproduces
R2=0.686, RMSE=12.23 NTU against Sentinel2_Extract_Ubon_New.csv (N=26 Ubon PCD
stations) when re-run fresh in this session.

Correction: an earlier message in this project claimed this same model reproduced
R2=0.653/RMSE=12.86 when paired with scaler_ubon.pkl. That claim was never actually
re-verified end-to-end and turned out to be wrong - scaler_ubon.pkl gives R2=-0.53
(badly wrong scaling) with this model. scaler_ubon_operational.pkl is the correct
pairing (same scaler cluster used by the Random Forest pipeline previously wired
into this dashboard, which the RF+calibrator combo reproduced almost exactly).

Also ruled out: model_neural_network_mlp.pkl (the other MLP file) gives a
suspicious R2 of ~1.0 or 0.96 on this same validation set - a sign it was trained
directly on these rows (data leakage), not a genuine held-out result. Not used here.
"""
import functools

import joblib
import numpy as np

FEATURES = ["B2", "B3", "B4", "B8", "NDWI", "MNDWI", "NDTI", "NDSSI"]

MODEL_PATH = "best_model_neural_network_mlp.pkl"
SCALER_PATH = "scaler_ubon_operational.pkl"
CALIBRATOR_PATH = "calibrator_neural_network_mlp.pkl"
MODEL_LABEL = "Neural Network / MLP (calibrated)"
VALIDATION_R2 = 0.686
VALIDATION_RMSE = 12.23
VALIDATION_N = 26


@functools.lru_cache(maxsize=1)
def load_pipeline():
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    return scaler, model, calibrator


def predict(feature_rows: np.ndarray) -> np.ndarray:
    """feature_rows: (N, 8) array in FEATURES order -> calibrated turbidity (NTU)."""
    scaler, model, calibrator = load_pipeline()
    scaled = scaler.transform(feature_rows)
    raw_pred = model.predict(scaled)
    calibrated = calibrator.predict(raw_pred.reshape(-1, 1))
    return np.clip(calibrated, a_min=0.0, a_max=None)


def compute_indices(b2, b3, b4, b8):
    def norm_diff(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (a - b) / (a + b)
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    ndwi = norm_diff(b3, b8)
    return ndwi


def compute_all_indices(b2, b3, b4, b8, b11):
    def norm_diff(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (a - b) / (a + b)
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    ndwi = norm_diff(b3, b8)
    mndwi = norm_diff(b3, b11)
    ndti = norm_diff(b4, b3)
    ndssi = norm_diff(b8, b4)
    return ndwi, mndwi, ndti, ndssi
