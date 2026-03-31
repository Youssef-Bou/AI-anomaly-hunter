from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model


@dataclass
class AnalysisConfig:
    model_path: Path = Path("TRAIN_model.keras")
    target_id: str = "KIC 8435766"
    data_points: int = 1000
    flatten_window_length: int = 401
    author: str = "Kepler"
    quarter: Optional[int] = 10
    save_plot: Optional[Path] = None
    log_level: str = "INFO"
    threshold_binary: float = 0.05
    threshold_planet: float = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse a star light curve using the trained anomaly-hunter autoencoder."
    )
    parser.add_argument("--model-path", default="TRAIN_model.keras")
    parser.add_argument("--target-id", default="KIC 8435766",
                        help="Star ID, e.g. 'KIC 8435766' or 'TIC 12345'.")
    parser.add_argument("--data-points", type=int, default=1000)
    parser.add_argument("--flatten-window-length", type=int, default=401)
    parser.add_argument("--author", default="Kepler")
    parser.add_argument("--quarter", type=int, default=10,
                        help="Kepler quarter / TESS sector. Pass 0 to omit.")
    parser.add_argument("--save-plot", default=None,
                        help="Save figure to this path instead of showing it.")
    parser.add_argument("--threshold-binary", type=float, default=0.05)
    parser.add_argument("--threshold-planet", type=float, default=0.001)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        model_path=Path(args.model_path),
        target_id=args.target_id,
        data_points=args.data_points,
        flatten_window_length=args.flatten_window_length,
        author=args.author,
        quarter=args.quarter if args.quarter != 0 else None,
        save_plot=Path(args.save_plot) if args.save_plot else None,
        threshold_binary=args.threshold_binary,
        threshold_planet=args.threshold_planet,
        log_level=args.log_level,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_trained_model(config: AnalysisConfig):
    if not config.model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {config.model_path}. "
            "Run the pipeline first to train a model."
        )
    logging.info("Loading model from %s", config.model_path)
    return load_model(config.model_path)


def download_light_curve(config: AnalysisConfig):
    logging.info("Searching for '%s' (author=%s, quarter=%s)",
                 config.target_id, config.author, config.quarter)
    kwargs: dict = {"author": config.author}
    if config.quarter is not None:
        kwargs["quarter"] = config.quarter

    search = lk.search_lightcurve(config.target_id, **kwargs)
    if len(search) == 0:
        raise ValueError(
            f"No light curve found for '{config.target_id}' "
            f"(author={config.author}, quarter={config.quarter})."
        )
    logging.info("Downloading first result.")
    return search.download()


def preprocess(lc, config: AnalysisConfig) -> np.ndarray:
    cleaned = lc.remove_nans().normalize()

    if len(cleaned.flux) < 10:
        raise ValueError(f"Too few data points after NaN removal: {len(cleaned.flux)}")

    window_length = config.flatten_window_length
    if window_length >= len(cleaned.flux):
        window_length = len(cleaned.flux) - 2
        if window_length % 2 == 0:
            window_length -= 1
        logging.warning("flatten window_length adjusted to %s.", window_length)
    if window_length < 5:
        raise ValueError("Light curve too short for flattening.")

    flattened = cleaned.flatten(window_length=window_length)
    x_old = np.linspace(0, 1, len(flattened.flux))
    x_new = np.linspace(0, 1, config.data_points)
    y_new = np.interp(x_new, x_old, flattened.flux.value).astype(np.float32)

    if not np.isfinite(y_new).all():
        raise ValueError("Preprocessed array contains non-finite values.")

    return y_new


def classify(max_dip: float, config: AnalysisConfig) -> tuple:
    if max_dip > config.threshold_binary:
        return (
            "DOPPELSTERN (Eclipsing Binary)",
            f"Einbruch {max_dip:.2%} — zu tief fuer einen Planeten-Transit.",
        )
    if max_dip > config.threshold_planet:
        return (
            "KANDIDAT (moeglicher Exoplanet)",
            f"Tiefe {max_dip:.2%} liegt im Bereich eines Planeten-Transits.",
        )
    return (
        "Wahrscheinlich Rauschen",
        f"Kein signifikanter Einbruch erkennbar (Tiefe {max_dip:.2%}).",
    )


def plot_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: AnalysisConfig,
    max_dip: float,
    mse_score: float,
    verdict: str,
) -> None:
    diff = np.abs(y_true - y_pred)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Anomaly Analysis: {config.target_id}", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(y_true, label="Echte Messdaten (NASA)", color="black", alpha=0.75, linewidth=0.8)
    ax1.plot(y_pred, label="KI-Rekonstruktion", color="crimson", linewidth=1.8)
    ax1.set_title(f"Max. Dip: {max_dip:.2%}  |  Urteil: {verdict}")
    ax1.set_ylabel("Normalisierter Flux")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.2)

    ax2 = axes[1]
    ax2.plot(diff, color="darkorange", label="Abweichung (Anomalie-Signal)", linewidth=0.9)
    ax2.fill_between(range(config.data_points), diff, color="darkorange", alpha=0.25)
    ax2.axhline(config.threshold_planet, color="green", linestyle="--", linewidth=1, alpha=0.6,
                label=f"Planeten-Schwelle ({config.threshold_planet:.1%})")
    ax2.axhline(config.threshold_binary, color="red", linestyle="--", linewidth=1, alpha=0.6,
                label=f"Binary-Schwelle ({config.threshold_binary:.1%})")
    ax2.set_title(f"Residuum  |  MSE Score: {mse_score:.6f}")
    ax2.set_xlabel("Zeit (Datenpunkte 0-1000)")
    ax2.set_ylabel("|Delta Flux|")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.2)

    plt.tight_layout()

    if config.save_plot is not None:
        config.save_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.save_plot, dpi=150, bbox_inches="tight")
        logging.info("Plot saved to %s", config.save_plot)
        plt.close(fig)
    else:
        plt.show()


def analyse(config: AnalysisConfig) -> int:
    try:
        model = load_trained_model(config)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    try:
        lc_raw = download_light_curve(config)
    except ValueError as exc:
        logging.error("%s", exc)
        return 1

    try:
        y_true = preprocess(lc_raw, config)
    except ValueError as exc:
        logging.error("Preprocessing failed: %s", exc)
        return 1

    y_pred = model.predict(y_true.reshape(1, config.data_points, 1), verbose=0).flatten()

    mse_score = float(np.mean(np.square(y_true - y_pred)))
    max_dip = float(1.0 - np.min(y_true))

    verdict, reason = classify(max_dip, config)

    sep = "-" * 40
    logging.info(sep)
    logging.info("Ziel:           %s", config.target_id)
    logging.info("MSE-Score:      %.6f", mse_score)
    logging.info("Max. Dip-Tiefe: %.2f%%", max_dip * 100)
    logging.info("URTEIL:         %s", verdict)
    logging.info("Grund:          %s", reason)
    logging.info(sep)

    plot_analysis(y_true, y_pred, config, max_dip, mse_score, verdict)
    return 0


def main() -> int:
    args = parse_args()
    config = build_config(args)
    setup_logging(config.log_level)
    return analyse(config)


if __name__ == "__main__":
    raise SystemExit(main())
