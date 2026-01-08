import pandas as pd
import lightkurve as lk
import numpy as np
import os
import warnings

# --- KONFIGURATION ---
CSV_DATEI = "keplerstellar.csv"      # Der Name deiner Datei von der NASA
AUSGABE_ORDNER = "training_data"  # Hier landen die bereinigten Dateien
ANZAHL_STERNE = 500                # Zum Testen erst mal 50, später auf 10000 erhöhen!
DATENPUNKTE = 1000                # Jeder Stern wird auf exakt 1000 Punkte skaliert (für die KI)

# Ordner erstellen, falls nicht vorhanden
if not os.path.exists(AUSGABE_ORDNER):
    os.makedirs(AUSGABE_ORDNER)

# Warnungen unterdrücken (Lightkurve meckert manchmal wegen Metadaten)
warnings.filterwarnings("ignore")

def process_lightcurve(lc):
    """
    Reinigt die Lichtkurve und bringt sie auf exakt DATENPUNKTE Länge.
    """
    # 1. NaN entfernen und Normalisieren
    lc = lc.remove_nans().normalize()
    
    # 2. Flatten (Langzeittrends/Rotation entfernen) - WICHTIG für Anomalie-Erkennung
    lc = lc.flatten(window_length=401)
    
    # 3. Interpolation (Damit alle Arrays exakt gleich lang sind für das neuronale Netz)
    x_alt = np.linspace(0, 1, len(lc.flux))
    x_neu = np.linspace(0, 1, DATENPUNKTE)
    y_neu = np.interp(x_neu, x_alt, lc.flux.value)
    
    return y_neu

# --- HAUPTPROGRAMM ---
print("Lese CSV-Datei ein...")
# Hinweis: comment='#' ignoriert die Header-Texte der NASA-Datei
df = pd.read_csv(CSV_DATEI, comment='#')

# Wir müssen herausfinden, wie die ID-Spalte heißt (tic_id für TESS, kepid für Kepler)
if 'tic_id' in df.columns:
    id_col = 'tic_id'
    mission_name = 'TESS'
    print(f"TESS-Daten erkannt ({len(df)} Einträge).")
elif 'kepid' in df.columns:
    id_col = 'kepid'
    mission_name = 'Kepler'
    print(f"Kepler-Daten erkannt ({len(df)} Einträge).")
else:
    raise ValueError("Konnte keine ID-Spalte (tic_id oder kepid) finden!")

# Sortieren nach CDPP (die ruhigsten zuerst), falls nicht schon passiert
if 'rrmscdpp06p0' in df.columns: # Der Name kann variieren, prüf deine CSV
    df = df.sort_values(by='rrmscdpp06p0', ascending=True)

print(f"Starte Download für die Top {ANZAHL_STERNE} Sterne...")

erfolg_count = 0

for index, row in df.head(ANZAHL_STERNE).iterrows():
    stern_id = int(row[id_col])
    target_name = f"{mission_name} {stern_id}"
    
    dateiname = os.path.join(AUSGABE_ORDNER, f"{stern_id}.npy")
    
    # Überspringen, wenn wir den schon haben
    if os.path.exists(dateiname):
        print(f"[{index}] Überspringe {stern_id} (schon vorhanden)")
        erfolg_count += 1
        continue

    try:
        # 1. Suche nach Daten
        # author="SPOC" bei TESS oder "Kepler" bei Kepler liefert die offiziellen Daten
        search = lk.search_lightcurve(target_name, author=("SPOC", "Kepler"))
        
        # Wir nehmen nur das erste gute Ergebnis (z.B. Sektor 1 oder Quarter 1)
        if len(search) > 0:
            lc = search[0].download()
            
            # 2. Verarbeiten
            clean_data = process_lightcurve(lc)
            
            # 3. Speichern (als effizientes NumPy Array)
            np.save(dateiname, clean_data)
            
            print(f"[{index}] Erfolg: {stern_id}")
            erfolg_count += 1
        else:
            print(f"[{index}] Keine Daten gefunden für {stern_id}")

    except Exception as e:
        print(f"[{index}] Fehler bei {stern_id}: {e}")

print(f"\nFertig! {erfolg_count} von {ANZAHL_STERNE} Sternen erfolgreich gespeichert.")