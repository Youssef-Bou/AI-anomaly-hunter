import pandas as pd

# 1. Lade deine riesige CSV-Tabelle
df = pd.read_csv("keplerstellar.csv")

# 2. Sortiere nach "Ruhe" (CDPP), damit die besten oben stehen
# (Achte darauf, wie die Spalte in deiner CSV genau heißt, oft 'rrmscdpp06p0')
df_sorted = df.sort_values(by='rrmscdpp06p0', ascending=True)

# 3. Nimm nur die besten 15.000
trainings_liste = df_sorted.head(15000)
print(trainings_liste[10:15])  # Zeige einige Beispiele aus der Liste
print(f"Wir nutzen nun {len(trainings_liste)} Sterne für das Training.")
# 4. Speichere die Trainingsliste in eine neue CSV-Datei
trainings_liste.to_csv("trainings_liste.csv", index=False)  
print("Trainingsliste gespeichert in 'trainings_liste.csv'.")