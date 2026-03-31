from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import lightkurve as lk
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Input, MaxPooling1D, UpSampling1D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


@dataclass
class PipelineConfig:
    train_csv: Path = Path("train_candidates.csv")
    search_csv: Path = Path("search_targets.csv")
    result_file: Path = Path("anomalie_ergebnisse.csv")
    train_dir: Path = Path("pipeline_train_data")
    model_path: Path = Path("TRAIN_model.keras")
    threshold_path: Path = Path("mse_threshold.npy")
    n_train_download: int = 2000
    n_search_analysis: int = 5000
    data_points: int = 1000
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.1
    min_training_samples: int = 50
    flatten_window_length: int = 401
    checkpoint_every: int = 20
    random_state: int = 42
    augment_data: bool = True
    log_level: str = "INFO"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run an anomaly-hunting autoencoder.")
    parser.add_argument("--phase", choices=["all", "harvest", "train", "hunt"], default="all")
    parser.add_argument("--train-csv", default="train_candidates.csv")
    parser.add_argument("--search-csv", default="search_targets.csv")
    parser.add_argument("--result-file", default="anomalie_ergebnisse.csv")
    parser.add_argument("--train-dir", default="pipeline_train_data")
    parser.add_argument("--model-path", default="TRAIN_model.keras")
    parser.add_argument("--n-train-download", type=int, default=2000)
    parser.add_argument("--n-search-analysis", type=int, default=5000)
    parser.add_argument("--data-points", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--flatten-window-length", type=int, default=401)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        train_csv=Path(args.train_csv),
        search_csv=Path(args.search_csv),
        result_file=Path(args.result_file),
        train_dir=Path(args.train_dir),
        model_path=Path(args.model_path),
        n_train_download=args.n_train_download,
        n_search_analysis=args.n_search_analysis,
        data_points=args.data_points,
        batch_size=args.batch_size,
        epochs=args.epochs,
        flatten_window_length=args.flatten_window_length,
        checkpoint_every=args.checkpoint_every,
        augment_data=not args.no_augment,
        log_level=args.log_level,
    )


def load_candidate_table(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, comment="#")
    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")
    return df


def detect_mission_and_id(df: pd.DataFrame) -> Tuple[str, str]:
    if "tic_id" in df.columns:
        return "TESS", "tic_id"
    if "kepid" in df.columns:
        return "Kepler", "kepid"
    raise ValueError("CSV must contain either 'tic_id' or 'kepid'.")


def preprocess_lc(lc, config: PipelineConfig) -> Optional[np.ndarray]:
    try:
        cleaned = lc.remove_nans().normalize()
        if len(cleaned.flux) < 10:
            logging.warning("Skipping light curve with too few samples: %s", len(cleaned.flux))
            return None

        window_length = min(config.flatten_window_length, len(cleaned.flux) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < 5:
            logging.warning("Skipping light curve because flatten window is too small.")
            return None

        flattened = cleaned.flatten(window_length=window_length)
        x_old = np.linspace(0, 1, len(flattened.flux))
        x_new = np.linspace(0, 1, config.data_points)
        y_new = np.interp(x_new, x_old, flattened.flux.value)

        if not np.isfinite(y_new).all():
            logging.warning("Skipping light curve due to non-finite values after interpolation.")
            return None
        return y_new.astype(np.float32)
    except Exception as exc:
        logging.exception("Preprocessing failed: %s", exc)
        return None


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random noise + time-shift augmentation to a batch (shape: N, T, 1)."""
    noise = rng.normal(0, 0.0005, x.shape).astype(np.float32)
    shift = int(rng.integers(-50, 51))
    return np.roll(x + noise, shift, axis=1)


def search_and_download(target: str):
    search_result = lk.search_lightcurve(target, author=("SPOC", "Kepler"))
    if len(search_result) == 0:
        logging.info("No light curve found for %s", target)
        return None
    return search_result[0].download()


def save_results_atomic(results: list, path: Path) -> None:
    if not results:
        return
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def run_harvest(config: PipelineConfig) -> bool:
    logging.info("Phase 1: HARVEST")
    config.train_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_candidate_table(config.train_csv)
        mission, id_col = detect_mission_and_id(df)
    except Exception as exc:
        logging.error("Harvest setup failed: %s", exc)
        return False

    downloaded = 0
    for _, row in df.iterrows():
        if downloaded >= config.n_train_download:
            break

        try:
            star_id = int(row[id_col])
        except Exception:
            logging.warning("Skipping row with invalid %s value: %s", id_col, row.get(id_col))
            continue

        save_path = config.train_dir / f"{star_id}.npy"
        if save_path.exists():
            downloaded += 1
            continue

        target = f"{mission} {star_id}"
        try:
            lc = search_and_download(target)
            if lc is None:
                continue
            data = preprocess_lc(lc, config)
            if data is None:
                continue
            np.save(save_path, data)
            downloaded += 1
            logging.info("[%s/%s] Downloaded %s", downloaded, config.n_train_download, target)
        except Exception as exc:
            logging.exception("Harvest failed for %s: %s", target, exc)

    logging.info("Harvest completed with %s prepared training samples.", downloaded)
    return downloaded >= config.min_training_samples


def load_training_arrays(config: PipelineConfig) -> np.ndarray:
    files = sorted(config.train_dir.glob("*.npy"))
    if len(files) < config.min_training_samples:
        raise ValueError(
            f"Not enough training samples in {config.train_dir}. "
            f"Found {len(files)}, need at least {config.min_training_samples}."
        )

    arrays = []
    for file in files:
        arr = np.load(file)
        if arr.shape != (config.data_points,):
            logging.warning("Skipping %s because shape is %s", file.name, arr.shape)
            continue
        if not np.isfinite(arr).all():
            logging.warning("Skipping %s because it contains non-finite values.", file.name)
            continue
        arrays.append(arr)

    if len(arrays) < config.min_training_samples:
        raise ValueError("Too many corrupted arrays were skipped.")

    x_train = np.asarray(arrays, dtype=np.float32)
    return x_train.reshape(len(x_train), config.data_points, 1)


def build_autoencoder(config: PipelineConfig) -> Model:
    """
    Improved Conv1D autoencoder:
    - 3 encoder stages (was 2) for a tighter bottleneck
    - BatchNormalization after each Conv1D for faster, more stable training
    - Wider kernels (7, 5, 3) to capture transit shapes at multiple scales
    - Linear output activation (was sigmoid) — correct for normalised flux near 1.0
    - Adam with lr=1e-3 (unchanged default, but now paired with ReduceLROnPlateau)
    """
    inputs = Input(shape=(config.data_points, 1))

    # --- Encoder ---
    x = Conv1D(64, 7, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding="same")(x)          # 500

    x = Conv1D(32, 5, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding="same")(x)          # 250

    x = Conv1D(16, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling1D(2, padding="same")(x)    # 125  ← bottleneck

    # --- Decoder ---
    x = Conv1D(16, 3, activation="relu", padding="same")(encoded)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)                          # 250

    x = Conv1D(32, 5, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)                          # 500

    x = Conv1D(64, 7, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)                          # 1000

    # Linear activation: normalised flux is NOT bounded to [0,1]
    outputs = Conv1D(1, 3, activation="linear", padding="same")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    model.summary(print_fn=logging.info)
    return model


def run_training(config: PipelineConfig) -> bool:
    logging.info("Phase 2: LEARN")
    try:
        dataset = load_training_arrays(config)
    except Exception as exc:
        logging.error("Training setup failed: %s", exc)
        return False

    x_train, x_val = train_test_split(
        dataset,
        test_size=config.validation_split,
        random_state=config.random_state,
        shuffle=True,
    )

    # --- Data augmentation: double the training set with jittered/shifted copies ---
    if config.augment_data:
        rng = np.random.default_rng(config.random_state)
        x_aug = augment_batch(x_train, rng)
        x_train = np.concatenate([x_train, x_aug], axis=0)
        logging.info("Augmentation enabled: training set expanded to %s samples.", len(x_train))

    model = build_autoencoder(config)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(config.model_path, monitor="val_loss", save_best_only=True, verbose=1),
        # Halve LR when val_loss stalls for 4 epochs; floor at 1e-6
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    logging.info("Starting training on %s samples (val: %s).", len(x_train), len(x_val))
    model.fit(
        x_train,
        x_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(x_val, x_val),
        callbacks=callbacks,
        verbose=1,
    )
    logging.info("Training completed. Best model saved to %s", config.model_path)

    # --- Dynamic threshold: 95th-percentile MSE on validation set ---
    try:
        val_pred = model.predict(x_val, verbose=0)
        val_mse = np.mean(np.square(x_val.squeeze() - val_pred.squeeze()), axis=1)
        threshold_95 = float(np.percentile(val_mse, 95))
        np.save(config.threshold_path, threshold_95)
        logging.info(
            "Auto-threshold (95th pct val MSE): %.6f  →  saved to %s",
            threshold_95,
            config.threshold_path,
        )
    except Exception as exc:
        logging.warning("Could not compute auto-threshold: %s", exc)

    return True


def run_hunt(config: PipelineConfig) -> bool:
    logging.info("Phase 3: HUNT")
    if not config.model_path.exists():
        logging.error("Model file not found: %s", config.model_path)
        return False

    # Load dynamic threshold if available, else fall back to a sensible default
    if config.threshold_path.exists():
        auto_threshold = float(np.load(config.threshold_path))
        logging.info("Using auto-threshold from training: %.6f", auto_threshold)
    else:
        auto_threshold = None
        logging.info("No auto-threshold found; using raw MSE scores only.")

    try:
        df_search = load_candidate_table(config.search_csv)
        mission, id_col = detect_mission_and_id(df_search)
        model = load_model(config.model_path)
    except Exception as exc:
        logging.error("Hunt setup failed: %s", exc)
        return False

    results = []
    checked_count = 0

    for _, row in df_search.head(config.n_search_analysis).iterrows():
        try:
            star_id = int(row[id_col])
        except Exception:
            logging.warning("Skipping row with invalid %s value: %s", id_col, row.get(id_col))
            continue

        target = f"{mission} {star_id}"
        try:
            lc = search_and_download(target)
            if lc is None:
                continue
            data = preprocess_lc(lc, config)
            if data is None:
                continue

            input_data = data.reshape(1, config.data_points, 1)
            reconstruction = model.predict(input_data, verbose=0).flatten()
            mse_score = float(np.mean(np.square(data - reconstruction)))

            is_anomaly = (auto_threshold is not None) and (mse_score > auto_threshold)
            results.append({
                "ID": star_id,
                "Score": mse_score,
                "Mission": mission,
                "Anomaly": is_anomaly,
                "Threshold": auto_threshold,
            })
            checked_count += 1
            flag = " *** ANOMALY ***" if is_anomaly else ""
            logging.info("%s -> score %.6f%s", target, mse_score, flag)

            if checked_count % config.checkpoint_every == 0:
                save_results_atomic(results, config.result_file)
                logging.info("Checkpoint written after %s analyzed targets.", checked_count)
        except Exception as exc:
            logging.exception("Hunt failed for %s: %s", target, exc)

    save_results_atomic(results, config.result_file)
    if not results:
        logging.warning("No hunt results were generated.")
        return False

    n_anomalies = sum(1 for r in results if r.get("Anomaly"))
    logging.info(
        "Hunt completed. %s targets checked, %s anomalies flagged. Results: %s",
        checked_count, n_anomalies, config.result_file,
    )
    return True


def main() -> int:
    start_time = time.time()
    args = parse_args()
    config = build_config(args)
    setup_logging(config.log_level)

    success = True
    if args.phase == "harvest":
        success = run_harvest(config)
    elif args.phase == "train":
        success = run_training(config)
    elif args.phase == "hunt":
        success = run_hunt(config)
    else:
        success = run_harvest(config) and run_training(config) and run_hunt(config)

    duration_hours = (time.time() - start_time) / 3600
    logging.info("Pipeline finished after %.2f hours.", duration_hours)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
