# AI Anomaly Hunter — Web Visualizer

Interaktive Web-App zur Echtzeit-Analyse beliebiger Sterne mit dem trainierten Convolutional Autoencoder.

## Voraussetzungen

1. `TRAIN_model.keras` im selben Ordner vorhanden (erzeugt durch `python TRAIN_HUNT_pipeline.py`)
2. Python 3.10+

## Installation & Start

```bash
git clone https://github.com/Youssef-Bou/AI-anomaly-hunter.git
cd AI-anomaly-hunter
pip install -r requirements.txt
python app.py
```

Dann im Browser oeffnen: **http://localhost:5050**

## Funktionsweise

1. Stern-ID eingeben (z.B. `KIC 8435766` oder `TIC 394137592`)
2. Mission (Kepler / TESS) und Quarter/Sector waehlen
3. "Analysieren" klicken

Das Backend (`app.py`) fuehrt exakt dieselbe Logik wie `check_anomaly.py` aus:
- `lightkurve.search_lightcurve()` — laedt NASA-Daten
- `remove_nans().normalize().flatten()` — Preprocessing
- `model.predict()` — Autoencoder-Inferenz
- MSE + Dip-Tiefe berechnen → Klassifizierung

Das Frontend zeigt:
- **Lichtkurve**: NASA-Daten (blau) vs. KI-Rekonstruktion (gelb)
- **Residuum-Plot**: Balken farbig nach Anomalie-Staerke (blau/gelb/rot)
- **Verdict-Karte**: Klassifizierung, MSE-Score, Dip-Tiefe, Intensitaets-Balken
- **Analyse-Verlauf**: Alle Sessions klickbar gespeichert

## Schwellenwerte

| Parameter | Default | Bedeutung |
|---|---|---|
| Planeten-Schwelle | 0.001 (0.1%) | Ab hier: Exoplanet-Kandidat |
| Binary-Schwelle | 0.05 (5%) | Ab hier: Eclipsing Binary |

Beide Werte sind im UI anpassbar (identisch zu `--threshold-planet` / `--threshold-binary` in `check_anomaly.py`).

## Beispiel-Sterne

| Stern-ID | Mission | Quarter | Erwartetes Ergebnis |
|---|---|---|---|
| KIC 8435766 | Kepler | 10 | Exoplanet-Kandidat |
| KIC 3114667 | Kepler | 6 | Kepler-78b (Planet) |
| KIC 8311864 | Kepler | 10 | Grenzfall / Rauschen |
| TIC 394137592 | TESS (SPOC) | 0 | TESS-Kandidat |

## API

```
POST /api/analyze
Content-Type: application/json

{
  "star_id": "KIC 8435766",
  "author": "Kepler",
  "quarter": 10,
  "threshold_planet": 0.001,
  "threshold_binary": 0.05
}
```

Antwort:
```json
{
  "star_id": "KIC 8435766",
  "real": [...],
  "reconstructed": [...],
  "residual": [...],
  "mse": 0.031452,
  "max_dip": 0.00182,
  "max_dip_pct": "0.18%",
  "verdict": "EXOPLANET-KANDIDAT",
  "verdict_type": "planet",
  "reason": "Tiefe 0.18% liegt im Bereich eines Planeten-Transits."
}
```
