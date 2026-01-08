import lightkurve as lk
import matplotlib.pyplot as plt

# BEISPIEL: Wir laden Daten von einem Stern (hier: Kepler-8)
# Für dein ML-Training würdest du später einfach über tausende IDs iterieren.

# 1. Suche nach Daten im Archiv (MAST)
# Wir suchen nach "Lichtkurven" (Helligkeit über Zeit) vom Autor "Kepler"
search_result = lk.search_lightcurve("Kepler-8", author="Kepler", quarter=10)

# 2. Download der Daten
lc = search_result.download()

# 3. Bereinigung (WICHTIG für Machine Learning!)
# Remove NaNs: Entfernt kaputte Datenpunkte
# Flatten: Entfernt den langfristigen Trend (Sternrotation), behält nur kurze Änderungen
lc_clean = lc.remove_nans().flatten(window_length=401)

# 4. Zugriff auf die rohen Zahlen für dein neuronales Netz
flux_daten = lc_clean.flux.value  # Das ist dein Input-Array (Y-Achse: Helligkeit)
zeit_daten = lc_clean.time.value  # Das ist die X-Achse (Zeit)

print(f"Anzahl der Datenpunkte: {len(flux_daten)}")
print(f"Beispiel-Werte: {flux_daten[:5]}")

# 5. Anzeigen
lc_clean.plot()
plt.show()