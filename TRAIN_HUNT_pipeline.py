import pandas as pd
import lightkurve as lk
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# =============================================================================
# KONFIGURATION & SETUP
# =============================================================================

# Dateinamen der Eingabe-Listen (CSV)
# TRAIN_CSV: Enthält ruhige Sterne (CDPP < 300) für das Training der "Normalität".
TRAIN_CSV = "train_candidates.csv"   

# SEARCH_CSV: Enthält ungefilterte Sterne, in denen wir Anomalien suchen.
SEARCH_CSV = "search_targets.csv"    

# Ausgabedatei für die gefundenen Kandidaten
RESULT_FILE = "anomalie_ergebnisse.csv"

# Pfade & Speicherorte
TRAIN_DIR = "pipeline_train_data"    # Temporärer Ordner für Trainingsdaten (.npy)
MODEL_PATH = "TRAIN_model.keras"     # Dateiname des trainierten Modells

# Steuerung der Datenmenge (Zeitmanagement)
# N_TRAIN_DOWNLOAD: Wie viele Sterne sollen gelernt werden? (ca. 1000 = 45 Min)
N_TRAIN_DOWNLOAD = 2000   

# N_SEARCH_ANALYSIS: Wie viele Sterne sollen durchsucht werden?
N_SEARCH_ANALYSIS = 5000  

# KI-Hyperparameter (Müssen exakt mit dem Analyse-Skript übereinstimmen!)
DATENPUNKTE = 1000      # Feste Länge jeder Lichtkurve
BATCH_SIZE = 32         # Wie viele Sterne lernt die KI gleichzeitig
EPOCHS = 100            # Maximale Anzahl der Trainingsrunden

# =============================================================================
# HILFSFUNKTIONEN (PREPROCESSING)
# =============================================================================

def preprocess_lc(lc):
    """
    Bereitet eine rohe Lichtkurve für das neuronale Netz auf.
    
    Schritte:
    1. Normalisierung: Flux-Werte um 1.0 zentrieren.
    2. Flattening: Entfernt natürliche Rotationstrends des Sterns.
    3. Interpolation: Zwingt die Daten auf exakt 1000 Punkte (Input-Vektor).
    """
    try:
        # 1. Bereinigen (NaNs entfernen) und Normalisieren
        lc = lc.remove_nans().normalize()
        
        # 2. Trendbereinigung (Wichtig für Anomalie-Erkennung)
        # window_length=401 glättet langsame Veränderungen.
        lc = lc.flatten(window_length=401)
        
        # 3. Interpolation auf feste Länge (DATENPUNKTE)
        x_alt = np.linspace(0, 1, len(lc.flux))
        x_neu = np.linspace(0, 1, DATENPUNKTE)
        y_neu = np.interp(x_neu, x_alt, lc.flux.value)
        
        return y_neu
    except:
        return None

# =============================================================================
# PHASE 1: HARVEST (DATEN BESCHAFFUNG)
# =============================================================================
def run_harvest():
    print(f"\n=== PHASE 1: HARVEST (Download von {N_TRAIN_DOWNLOAD} Trainings-Sternen) ===")
    
    # Sicherstellen, dass der Speicherordner existiert
    if not os.path.exists(TRAIN_DIR): 
        os.makedirs(TRAIN_DIR)
    
    # CSV-Liste laden
    try:
        df = pd.read_csv(TRAIN_CSV, comment='#')
    except FileNotFoundError:
        print(f"FEHLER: Datei {TRAIN_CSV} nicht gefunden! Bitte erstellen.")
        return False

    # Automatische Erkennung: Kepler (kepid) oder TESS (tic_id)?
    id_col = 'tic_id' if 'tic_id' in df.columns else 'kepid'
    mission = "TESS" if 'tic_id' in df.columns else "Kepler"
    print(f"Modus erkannt: {mission} (ID-Spalte: {id_col})")
    
    count = 0
    downloaded = 0
    
    # Iteration durch die Kandidaten-Liste
    for index, row in df.iterrows():
        # Abbruchbedingung erreicht?
        if downloaded >= N_TRAIN_DOWNLOAD: 
            break
        
        stern_id = int(row[id_col])
        save_path = os.path.join(TRAIN_DIR, f"{stern_id}.npy")
        
        # Effizienz: Überspringen, wenn wir den Stern schon haben
        if os.path.exists(save_path):
            downloaded += 1
            continue
            
        try:
            target = f"{mission} {stern_id}"
            # Suche im NASA Archiv ("SPOC" = offizielle TESS Pipeline)
            search = lk.search_lightcurve(target, author=("SPOC", "Kepler"))
            
            if len(search) > 0:
                # Download des ersten verfügbaren Datensatzes
                lc = search[0].download()
                data = preprocess_lc(lc)
                
                if data is not None:
                    # Speichern als effizientes NumPy Array
                    np.save(save_path, data)
                    print(f"[{downloaded+1}/{N_TRAIN_DOWNLOAD}] Download erfolgreich: {stern_id}")
                    downloaded += 1
            else:
                print(f"Keine Daten gefunden für {stern_id}")
                
        except Exception as e:
            print(f"Fehler bei {stern_id}: {e}")
            
    print(f"Phase 1 abgeschlossen. {downloaded} Trainings-Dateien bereit.")
    return True

# =============================================================================
# PHASE 2: LEARN (TRAINING DES MODELLS)
# =============================================================================
def run_training():
    print("\n=== PHASE 2: LEARN (Training des Autoencoders) ===")
    
    # 1. Daten in den Arbeitsspeicher laden
    dateien = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.npy')]
    
    if len(dateien) < 50:
        print("FEHLER: Zu wenig Trainingsdaten! Pipeline bricht ab.")
        return False

    print(f"Lade {len(dateien)} Sterne in den Speicher...")
    data_list = []
    for f in dateien:
        data_list.append(np.load(os.path.join(TRAIN_DIR, f)))
    
    # Umwandlung in TensorFlow-kompatibles Format (Samples, 1000, 1)
    x_train = np.array(data_list)
    x_train = np.reshape(x_train, (len(x_train), DATENPUNKTE, 1))
    
    # Split: 90% Training, 10% Validierung (Test während des Trainings)
    x_train, x_test = train_test_split(x_train, test_size=0.1, random_state=42)

    # 2. Architektur des Convolutional Autoencoder
    input_img = Input(shape=(DATENPUNKTE, 1))
    
    # -- ENCODER (Komprimierung) --
    # Findet Muster und reduziert die Daten auf das Wesentliche
    x = Conv1D(32, 3, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x) 
    
    # -- DECODER (Rekonstruktion) --
    # Versucht, aus den komprimierten Daten den ursprünglichen Stern zu malen
    x = Conv1D(16, 3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    # Modell kompilieren
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse') # MSE = Mean Squared Error

    # 3. Callbacks (Intelligentes Training)
    callbacks = [
        # EarlyStopping: Stoppt, wenn das Modell nicht mehr dazulernt (spart Zeit)
        EarlyStopping(monitor='val_loss', patience=8, verbose=1),
        # ModelCheckpoint: Speichert immer die beste Version des Modells
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # 4. Training starten
    print("Starte neuronales Training...")
    autoencoder.fit(x_train, x_train,  # Input == Output (Autoencoder Prinzip)
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    validation_data=(x_test, x_test),
                    callbacks=callbacks,
                    verbose=1)
                    
    print(f"Training fertig. Bestes Modell gespeichert unter '{MODEL_PATH}'.")
    return True

# =============================================================================
# PHASE 3: HUNT (SUCHE NACH ANOMALIEN)
# =============================================================================
def run_hunt():
    print(f"\n=== PHASE 3: THE HUNT (Analyse von {N_SEARCH_ANALYSIS} neuen Sternen) ===")
    
    # Prüfen, ob Modell existiert
    if not os.path.exists(MODEL_PATH):
        print("FEHLER: Kein Modell gefunden! Training fehlgeschlagen?")
        return

    # Modell laden
    model = load_model(MODEL_PATH)
    
    # Such-Liste laden
    try:
        df_search = pd.read_csv(SEARCH_CSV, comment='#')
    except FileNotFoundError:
        print(f"FEHLER: {SEARCH_CSV} nicht gefunden!")
        return

    id_col = 'tic_id' if 'tic_id' in df_search.columns else 'kepid'
    mission = "TESS" if 'tic_id' in df_search.columns else "Kepler"
    
    results = []
    checked_count = 0
    
    print("Starte Inferenz (Vorhersage)...")
    
    # Iteration durch die Such-Liste
    for index, row in df_search.head(N_SEARCH_ANALYSIS).iterrows():
        stern_id = int(row[id_col])
        target = f"{mission} {stern_id}"
        
        try:
            # 1. Live Download (Wir speichern NICHT, um Speicherplatz zu sparen)
            search = lk.search_lightcurve(target, author=("SPOC", "Kepler"))
            if len(search) == 0: continue
            
            lc = search[0].download()
            data = preprocess_lc(lc)
            
            if data is None: continue
            
            # 2. KI Vorhersage (Inferenz)
            input_data = np.reshape(data, (1, DATENPUNKTE, 1))
            reconstruction = model.predict(input_data, verbose=0)
            
            # 3. Score Berechnung (MSE)
            # Hoher Score = Die KI konnte den Stern nicht rekonstruieren -> ANOMALIE!
            mse_score = np.mean(np.square(data - reconstruction.flatten()))
            
            print(f"Stern {stern_id} -> Score: {mse_score:.6f}")
            
            # 4. Ergebnis merken
            results.append({
                "ID": stern_id,
                "Score": mse_score,
                "Mission": mission
            })
            
            checked_count += 1
            
            # Sicherheits-Speicherung alle 20 Sterne (falls PC abstürzt)
            if checked_count % 20 == 0:
                pd.DataFrame(results).to_csv(RESULT_FILE, index=False)
                
        except Exception as e:
            print(f"Überspringe {stern_id} wegen Fehler: {e}")

    # Endergebnis sortieren & speichern
    final_df = pd.DataFrame(results)
    if not final_df.empty:
        # Sortieren: Die größten Anomalien (höchster Score) nach oben!
        final_df = final_df.sort_values(by="Score", ascending=False)
        final_df.to_csv(RESULT_FILE, index=False)
        print(f"\nFERTIG! Ergebnisse erfolgreich in '{RESULT_FILE}' gespeichert.")
        print("Tipp: Öffne diese Datei und prüfe die Top-Kandidaten mit 'check_anomaly.py'!")
    else:
        print("Keine Ergebnisse gefunden.")

# =============================================================================
# HAUPTPROGRAMM (PIPELINE STEUERUNG)
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Die Pipeline führt die Phasen nacheinander aus
    # Wenn eine Phase fehlschlägt (False zurückgibt), stoppt die Pipeline.
    
    if run_harvest():      # Schritt 1: Daten laden
        if run_training(): # Schritt 2: KI trainieren
            run_hunt()     # Schritt 3: Anomalien suchen
    
    duration = (time.time() - start_time) / 3600
    print(f"\nPipeline beendet nach {duration:.2f} Stunden. Gute Nacht!")
