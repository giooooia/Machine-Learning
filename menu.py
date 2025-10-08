import os
import pandas as pd
from Data.dataset import Dataset


def carica_dataset(percorso=None):
    print("=== CARICAMENTO DATASET ===")

    if percorso is None:
        percorso = input("Inserisci il percorso del file CSV (es. data/dataset.csv): ")

    # Controllo che il file esista
    if not os.path.exists(percorso):
        print(f"Errore: il file '{percorso}' non esiste.")
        return None

    # Prova a leggere il CSV
    try:
        df = Dataset(str(percorso))
        print("Dataset caricato correttamente!")
        return df
    except Exception as e:
        print(f"Errore nel caricamento: {e}")
        return None
    
    
def analizza_dati(df):
    """
    Mostra un menu per analizzare i dati e chiama la funzione appropriata
    in base alla scelta dell'utente.
    """
    while True:
        print("\n=== MENU ANALISI DATI ===")
        print("1. Statistiche generali")
        print("2. Analisi univariata")
        print("3. Analisi bivariata")
        print("4. Analisi multivariata")
        print("0. Esci")

        scelta = input("Seleziona un'opzione: ").strip()

        if scelta == "0":
            print("Uscita dal menu di analisi.")
            break

        elif scelta == "1":
            print("\nSTATISTICHE GENERALI")
            statistiche_generali(df)

        elif scelta == "2":
            print("\nANALISI UNIVARIATA")
            analisi_univariata(df)

        elif scelta == "3":
            print("\nANALISI BIVARIATA")
            analisi_bivariata(df)

        elif scelta == "4":
            print("\nANALISI MULTIVARIATA")
            analisi_multivariata(df)

        else:
            print("Scelta non valida. Riprova.")
