"""
Zentrale Konfiguration für die AI Anomaly Hunter Pipeline.
Alle Parameter können hier angepasst werden.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    # --- Dateipfade ---
    train_csv: Path = Path("train_candidates.csv")
    search_csv: Path = Path("search_targets.csv")
    result_file: Path = Path("anomalie_ergebnisse.csv")
    train_dir: Path = Path("pipeline_train_data")
    model_path: Path = Path("TRAIN_model.keras")

    # --- Datenmenge ---
    n_train_download: int = 2000
    n_search_analysis: int = 5000

    # --- Modell-Hyperparameter ---
    datenpunkte: int = 1000      # Feste Länge jeder Lichtkurve
    batch_size: int = 32
    epochs: int = 100
    val_split: float = 0.1       # 10% Validierungsdaten
    early_stopping_patience: int = 8
    flatten_window: int = 401    # Muss ungerade sein


# Globale Standardkonfiguration
DEFAULT_CONFIG = PipelineConfig()
