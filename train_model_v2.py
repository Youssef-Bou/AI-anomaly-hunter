import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # <--- NEU: Wächter
from sklearn.model_selection import train_test_split

# --- KONFIGURATION ---
DATEN_ORDNER = "pipeline_train_data"
DATENPUNKTE = 1000  
BATCH_SIZE = 64      # <--- NEU: Höher für stabileres Lernen bei 500 Daten
EPOCHS = 10000        # <--- NEU: Wir geben ihm mehr Zeit, brechen aber ggf. früher ab

# 1. Daten laden (wie gehabt)
print("Lade Daten...")
dateien = [f for f in os.listdir(DATEN_ORDNER) if f.endswith('.npy')]
daten_liste = []

# Sicherheitscheck: Haben wir wirklich Daten?
if not dateien:
    print("FEHLER: Keine .npy Dateien gefunden!")
    exit()

for datei in dateien:
    pfad = os.path.join(DATEN_ORDNER, datei)
    array = np.load(pfad)
    daten_liste.append(array)

x_train = np.array(daten_liste)
x_train = np.reshape(x_train, (len(x_train), DATENPUNKTE, 1))

# Split: 80% Training, 20% Validierung (Test)
x_train, x_test = train_test_split(x_train, test_size=0.2, random_state=42)

print(f"Training mit {len(x_train)} Sternen, Validierung mit {len(x_test)} Sternen.")

# 2. Das verbesserte KI-Modell
input_img = Input(shape=(DATENPUNKTE, 1))

# Encoder (Mehr Filter: 32 statt 16)
x = Conv1D(32, 3, activation='relu', padding='same')(input_img) # <--- Mehr Kapazität
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
encoded = MaxPooling1D(2, padding='same')(x) 

# Decoder
x = Conv1D(16, 3, activation='relu', padding='same')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(32, 3, activation='relu', padding='same')(x) # <--- Symmetrisch zum Encoder
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 3. Die Wächter (Callbacks) einrichten
callbacks = [
    # Stoppt, wenn der 'val_loss' sich 10 Runden lang nicht verbessert
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    
    # Speichert NUR, wenn das Modell besser ist als das vorherige
    ModelCheckpoint('night_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# 4. Training
print("Starte Training...")
history = autoencoder.fit(
    x_train, x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=callbacks, # <--- Hier aktivieren wir die Wächter
    verbose=1
)

# 5. Visualisierung des Trainings-Erfolgs (Loss-Kurve)
# Das ist wichtig, um zu sehen, ob die KI wirklich lernt
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Trainings-Fehler')
plt.plot(history.history['val_loss'], label='Test-Fehler')
plt.title('Lernkurve (Je tiefer, desto besser)')
plt.ylabel('MSE Fehler')
plt.xlabel('Epoche')
plt.legend()
plt.show()

print("Fertig. Das beste Modell wurde als 'night_model.keras' gespeichert.")