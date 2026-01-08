import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =============================================================================
# KONFIGURATION
# =============================================================================
# Pfad zum trainierten Modell (muss mit der Pipeline übereinstimmen)
MODELL_DATEI = "TRAIN_model.keras"

# Die ID des Sterns, den wir untersuchen wollen (z.B. KIC 8311864)
TARGET_ID = "KIC 8435766"

# Die Länge der Lichtkurve (Muss exakt dem Wert beim Training entsprechen!)
DATENPUNKTE = 1000

# =============================================================================
# SCHRITT 1: MODELL & DATEN LADEN
# =============================================================================
print(f"Lade trainiertes Gehirn: {MODELL_DATEI}...")
try:
    model = load_model(MODELL_DATEI)
except OSError:
    print(f"FEHLER: Das Modell '{MODELL_DATEI}' wurde nicht gefunden.")
    exit()

print(f"Lade Daten für {TARGET_ID} von der NASA...")
# Wir suchen im NASA-Archiv nach Lichtkurven des Sterns
# author="Kepler" für Kepler-Daten, author="SPOC" für TESS-Daten
search = lk.search_lightcurve(TARGET_ID, author="Kepler", quarter=10)

if len(search) == 0:
    print("Fehler: Stern nicht gefunden oder keine Daten verfügbar!")
    exit()

# Download der Lichtkurve (Standardmäßig das erste Ergebnis)
lc_raw = search.download()

# =============================================================================
# SCHRITT 2: PREPROCESSING (DATENAUFBEREITUNG)
# =============================================================================
# Dieser Prozess muss 1:1 identisch mit dem Training sein, damit die KI
# die Daten versteht.

# 1. Grundreinigung: Entfernen von ungültigen Werten (NaN) und Normalisierung
# (Normalisierung setzt den Median-Flux auf 1.0)
lc = lc_raw.remove_nans().normalize()

# 2. Flattening: Entfernt langfristige Trends wie die Rotation des Sterns.
# window_length=401 ist der Standardwert, den wir im Training genutzt haben.
lc = lc.flatten(window_length=401)

# 3. Interpolation: Wir zwingen die Daten auf exakt 1000 Punkte.
# Egal wie lang die echte Messung war, die KI erwartet immer 1000 Inputs.
x_alt = np.linspace(0, 1, len(lc.flux))       # Alte Zeitachse
x_neu = np.linspace(0, 1, DATENPUNKTE)        # Neue Zeitachse (0 bis 1 in 1000 Schritten)
y_neu = np.interp(x_neu, x_alt, lc.flux.value) # Mapping der Helligkeitswerte

# 4. Reshaping für TensorFlow
# Das Modell erwartet die Form (Anzahl_Samples, Länge, Kanäle)
# -> (1 Stern, 1000 Datenpunkte, 1 Farbkanal/Helligkeit)
input_data = np.reshape(y_neu, (1, DATENPUNKTE, 1))

# =============================================================================
# SCHRITT 3: ANALYSE DURCH DIE KI (INFERENZ)
# =============================================================================
print("KI analysiert den Stern...")

# Die KI versucht, die Kurve zu rekonstruieren.
# Da sie nur "langweilige" Sterne gelernt hat, wird sie Anomalien (Transits)
# nicht zeichnen können. Die Differenz ist unser Signal.
rekonstruktion = model.predict(input_data, verbose=0)
rekonstruktion = rekonstruktion.flatten() # Zurückwandeln in flache Liste

# =============================================================================
# SCHRITT 4: FEHLER-BERECHNUNG & DIAGNOSE
# =============================================================================

# Berechnung des Mean Squared Error (MSE)
# Je höher dieser Wert, desto weniger sieht der Stern aus wie ein "normaler" Stern.
fehler = np.mean(np.square(y_neu - rekonstruktion))
print(f"ANOMALIE-SCORE (MSE): {fehler:.6f}")

# Automatische Diagnose basierend auf der Tiefe des Dips
min_flux = np.min(y_neu)      # Der dunkelste Punkt in der Kurve
max_dip = 1.0 - min_flux      # Wie viel Prozent des Lichts fehlen? (z.B. 0.01 = 1%)

print("-" * 30)
print(f"Tiefster Punkt: {min_flux:.4f}")
print(f"Maximale Tiefe: {max_dip:.2%}")

# Entscheidungslogik für die Klassifizierung
if max_dip > 0.05: # Tiefer als 5%
    print("URTEIL: Wahrscheinlich DOPPELSTERN (Eclipsing Binary)")
    print("Grund: Der Helligkeitseinbruch ist zu massiv für einen Planeten.")
    
elif max_dip > 0.001: # Zwischen 0.1% und 5%
    print("URTEIL: KANDIDAT! (Möglicher Exoplanet oder Brauner Zwerg)")
    print("Grund: Die Tiefe liegt im typischen Bereich für Planeten-Transits.")
    
else:
    print("URTEIL: Wahrscheinlich nur Rauschen")
    print("Grund: Keine signifikante Verdunkelung erkannt.")
print("-" * 30)

# =============================================================================
# SCHRITT 5: VISUALISIERUNG
# =============================================================================
# Wir erstellen zwei Grafiken untereinander

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Die Lichtkurven im Vergleich
ax1.plot(y_neu, label='Echte Messdaten (NASA)', color='black', alpha=0.7)
ax1.plot(rekonstruktion, label='KI-Rekonstruktion (Erwartung)', color='red', linewidth=2)
ax1.set_title(f"Analyse von {TARGET_ID} (Tiefe: {max_dip:.2%})")
ax1.set_ylabel("Normalisierter Flux")
ax1.legend()

# Plot 2: Das Differenz-Signal (Residuum)
# Hier sieht man genau, WO die Anomalie auftritt
differenz = np.abs(y_neu - rekonstruktion)
ax2.plot(differenz, color='orange', label='Abweichung (Anomalie-Signal)')
ax2.fill_between(range(DATENPUNKTE), differenz, color='orange', alpha=0.3)

# Hilfslinie bei 1% (typische Obergrenze für Gasriesen)
ax2.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Planeten-Limit (~1%)')

ax2.set_title(f"Anomalie-Signal (MSE Score: {fehler:.6f})")
ax2.set_xlabel("Zeit (Normalisierte Datenpunkte 0-1000)")
ax2.legend()

plt.tight_layout()
plt.show()