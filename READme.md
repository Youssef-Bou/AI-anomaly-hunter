# Deep Learning Anomaly Hunter

Ein KI-gestuetztes Tool zur automatischen Erkennung von Exoplaneten und astrophysikalischen Anomalien in Kepler- und TESS-Lichtkurven.

## Ueber das Projekt
Dieses Projekt nutzt einen Convolutional Autoencoder, um Anomalien in Zeitreihendaten der NASA zu finden. Das Modell lernt das "normale" Rauschen von Sternen und schlaegt Alarm, wenn untypische Muster (wie Transits oder Eclipses) auftreten.

Kernfunktionen:
* Unsupervised Learning: Kein gelabelter Datensatz notwendig.
* Automatische Anomalie-Erkennung: Unterscheidung zwischen Rauschen, Doppelsternen und potenziellen Exoplaneten.
* Validierung: Erfolgreiche Re-Identifikation von Kepler-78b und diversen Eclipsing Binaries.

## Struktur
* TRAIN_HUNTER_pipeline.py: Die Haupt-Pipeline. Fuehrt automatisiert den Download von Trainingsdaten, das Modell-Training und die Suche (Inferenz) durch.
* check_anomaly.py: Analyse-Tool zur Visualisierung gefundener Kandidaten und Berechnung der Dip-Tiefe.
* train_model_v2.py: Skript fuer manuelles Modell-Training und Hyperparameter-Tuning.

## Installation & Nutzung

1. Repository klonen:
   git clone https://github.com/DEIN_USERNAME/anomaly-hunter.git
   cd anomaly-hunter

2. Abhaengigkeiten installieren:
   pip install -r requirements.txt

3. Daten vorbereiten:
   Lade die Datei 'keplerstellar.csv' vom NASA Exoplanet Archive herunter.
   Erstelle daraus zwei Dateien im Projektordner:
   - 'train_candidates.csv' (fuer das Training, z.B. ruhige Sterne)
   - 'search_targets.csv' (fuer die Suche, ungefilterte Sterne)

4. Pipeline starten:
   python TRAIN_HUNTER_pipeline.py

## Ergebnisse interpretieren
Das Modell berechnet einen Rekonstruktionsfehler (MSE). Das Analyse-Skript klassifiziert Kandidaten basierend auf der Tiefe des Helligkeitseinbruchs:
* Tiefe < 1%: Moeglicher Exoplanet-Kandidat.
* Tiefe > 1-5%: Wahrscheinlich Bedeckungsveraenderlicher Stern (Eclipsing Binary).

---
Created with Python, TensorFlow & Lightkurve.