# Deep Learning Anomaly Hunter

Ein KI-gestuetztes Tool zur automatischen Erkennung von Exoplaneten und astrophysikalischen Anomalien in Kepler- und TESS-Lichtkurven.

---

## Ueber das Projekt

Dieses Projekt nutzt einen **Convolutional Autoencoder**, um Anomalien in Zeitreihendaten der NASA zu finden. Das Modell lernt das normale Helligkeitsrauschen von Sternen und schlaegt Alarm, wenn untypische Muster wie Transits oder Eclipses auftreten.

**Kernfunktionen:**
- **Unsupervised Learning** -- kein gelabelter Datensatz notwendig
- **Automatische Anomalie-Erkennung** -- unterscheidet Rauschen, Doppelsterne und Exoplanet-Kandidaten
- **Validierung** -- erfolgreiche Re-Identifikation von Kepler-78b und diversen Eclipsing Binaries
- **CLI-Steuerung** -- alle Parameter sind per Kommandozeile konfigurierbar

---

## Dateistruktur

```
AI-anomaly-hunter/
├── TRAIN_HUNT_pipeline.py       # Haupt-Pipeline (Harvest -> Train -> Hunt)
├── check_anomaly.py             # Analyse & Visualisierung einzelner Sterne
├── download_data.py             # Hilfsskript fuer manuellen Daten-Download
├── requirements.txt             # Python-Abhaengigkeiten
├── .gitignore
└── Kepler-78b_WIKI.png          # Referenzbild Kepler-78b (Wikipedia)
```

> **Hinweis:** Das trainierte Modell (`TRAIN_model.keras`) wird **nicht** im Repository versioniert.
> Es wird lokal durch Ausfuehren der Pipeline erzeugt.

---

## Installation

```bash
git clone https://github.com/Youssef-Bou/AI-anomaly-hunter.git
cd AI-anomaly-hunter
pip install -r requirements.txt
```

---

## Nutzung

### Schritt 1 -- Daten vorbereiten

Lade `keplerstellar.csv` vom [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) herunter und erstelle daraus zwei Dateien:

| Datei | Inhalt |
|---|---|
| `train_candidates.csv` | Ruhige Sterne (z. B. CDPP < 300) fuer das Training |
| `search_targets.csv` | Ungefilterte Sterne fuer die Anomalie-Suche |

Beide Dateien benoetigen eine Spalte `kepid` (Kepler) oder `tic_id` (TESS).

### Schritt 2 -- Pipeline starten

Die Pipeline kann als Ganzes oder phasenweise ausgefuehrt werden:

```bash
# Komplette Pipeline (Harvest -> Training -> Hunt)
python TRAIN_HUNT_pipeline.py

# Nur einzelne Phasen
python TRAIN_HUNT_pipeline.py --phase harvest --n-train-download 500
python TRAIN_HUNT_pipeline.py --phase train   --epochs 50
python TRAIN_HUNT_pipeline.py --phase hunt    --n-search-analysis 1000
```

**Alle Parameter im Ueberblick:**

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--phase` | `all` | `all`, `harvest`, `train` oder `hunt` |
| `--train-csv` | `train_candidates.csv` | Trainings-Kandidaten |
| `--search-csv` | `search_targets.csv` | Such-Kandidaten |
| `--n-train-download` | `2000` | Anzahl Trainingssterne (ca. 45 Min/1000) |
| `--n-search-analysis` | `5000` | Anzahl Sterne fuer die Suche |
| `--epochs` | `100` | Max. Trainingsepochen (EarlyStopping aktiv) |
| `--batch-size` | `32` | Batch-Groesse beim Training |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING` |

### Schritt 3 -- Kandidaten analysieren

```bash
# Einzelnen Stern analysieren (Plot wird angezeigt)
python check_anomaly.py --target-id "KIC 8435766"

# Plot als PNG speichern
python check_anomaly.py --target-id "KIC 8435766" --save-plot plots/KIC8435766.png

# TESS-Stern analysieren
python check_anomaly.py --target-id "TIC 394137592" --author SPOC --quarter 0

# Klassifizierungsschwellen anpassen
python check_anomaly.py --target-id "KIC 8311864" --threshold-planet 0.005 --threshold-binary 0.03
```

---

## Ergebnisse interpretieren

Das Modell berechnet einen **Rekonstruktionsfehler (MSE)**. Der Analyse-Plot zeigt:

1. **Oben:** Echte NASA-Messdaten vs. KI-Rekonstruktion
2. **Unten:** Residuum -- je groesser der Ausschlag, desto auffaelliger der Stern

**Automatische Klassifizierung nach Dip-Tiefe:**

| Tiefe des Einbruchs | Urteil |
|---|---|
| < 0,1 % | Wahrscheinlich Rauschen |
| 0,1 % -- 5 % | **Exoplanet-Kandidat** |
| > 5 % | Bedeckungsveraenderlicher Stern (Eclipsing Binary) |

---

## Architektur -- Convolutional Autoencoder

```
Input (1000, 1)
    |
    v
Conv1D(32, 3, relu) -> MaxPool1D(2)
Conv1D(16, 3, relu) -> MaxPool1D(2)
    |
 [Latenter Raum: 250 x 16]
    |
    v
Conv1D(16, 3, relu) -> UpSampling1D(2)
Conv1D(32, 3, relu) -> UpSampling1D(2)
Conv1D(1,  3, sigmoid)
    |
    v
Output (1000, 1)
```

**Trainingsidee:** Input == Output (Autoencoder-Prinzip). Das Netz lernt nur ruhige Sterne zu rekonstruieren. Ein Transit erzeugt einen hohen MSE -- Anomalie erkannt.

---

## Abhaengigkeiten

```
lightkurve
tensorflow
scikit-learn
pandas
numpy
matplotlib
```

---

*Created with Python, TensorFlow & Lightkurve.*
