import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from sklearn.model_selection import train_test_split

# --- KONFIGURATION ---
DATEN_ORDNER = "training_data"
DATENPUNKTE = 1000  # Muss exakt übereinstimmen mit dem Download-Skript!

# 1. Daten laden
print("Lade Daten...")
dateien = [f for f in os.listdir(DATEN_ORDNER) if f.endswith('.npy')]
daten_liste = []

for datei in dateien:
    pfad = os.path.join(DATEN_ORDNER, datei)
    array = np.load(pfad)
    daten_liste.append(array)

# Umwandeln in ein großes NumPy Array für TensorFlow
# Form: (Anzahl_Sterne, 1000, 1) -> Die 1 am Ende ist wichtig für Conv1D
x_train = np.array(daten_liste)
x_train = np.reshape(x_train, (len(x_train), DATENPUNKTE, 1))

# Aufteilen: Wir nutzen 90% zum Trainieren, 10% zum Prüfen
x_train, x_test = train_test_split(x_train, test_size=0.1, random_state=42)

print(f"Training mit {len(x_train)} Sternen, Test mit {len(x_test)} Sternen.")

# 2. Das KI-Modell bauen (Autoencoder)
input_img = Input(shape=(DATENPUNKTE, 1))

# --- ENCODER (Komprimieren) ---
x = Conv1D(16, 3, activation='relu', padding='same')(input_img)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(8, 3, activation='relu', padding='same')(x)
encoded = MaxPooling1D(2, padding='same')(x) 
# Jetzt sind die Daten stark komprimiert (Latent Space)

# --- DECODER (Wiederherstellen) ---
x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse') # MSE = Mean Squared Error (Fehlerquadrat)

# 3. Training
print("Starte Training...")
history = autoencoder.fit(x_train, x_train, # Input ist gleich Output!
                epochs=100,                  # Wie oft er übt
                batch_size=32,               # Bei nur 35 Dateien kleine Batch-Size wählen
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=1)

# 4. Ergebnis prüfen (Visualisierung)
print("Training beendet. Erstelle Test-Plot...")

# Wir lassen die KI mal einen Stern aus dem Test-Set malen
decoded_imgs = autoencoder.predict(x_test)

# Plotten: Original vs. Rekonstruktion
idx = 0 # Wir schauen uns den ersten Test-Stern an
plt.figure(figsize=(10, 4))

# Original
plt.subplot(1, 2, 1)
plt.plot(x_test[idx].flatten(), label='Original (Echt)')
plt.title("Was die KI gesehen hat")
plt.legend()

# Rekonstruktion
plt.subplot(1, 2, 2)
plt.plot(decoded_imgs[idx].flatten(), color='red', label='Rekonstruktion (KI)')
plt.title("Was die KI daraus gemalt hat")
plt.legend()

plt.show()

# Speichern des Modells für später
autoencoder.save("mein_anomaly_hunter.keras")
print("Modell gespeichert als 'mein_anomaly_hunter.keras'")