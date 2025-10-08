from Data.dataset import Dataset
import os

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
    in base alla scelta dell'utente. Prima del menu chiede se utilizzare
    il dataset preprocessato.
    """

    # Chiede se usare il dataset preprocessato
    usa_preprocessato = input("Vuoi eseguire l'analisi sul dataset preprocessato? (s/n): ").strip().lower()

    if usa_preprocessato == "s":
        print("Eseguo il preprocessing del dataset...")
        df = df.preprocessing()
        print("Preprocessing completato.\n")

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
            print("="*80)
            print("ANALISI PRELIMINARE")
            print("="*80)
            statistiche_generali(df)

        elif scelta == "2":
            print("="*80)
            print("ANALISI UNIVARIATA")
            print("="*80)
            analisi_univariata(df)

        elif scelta == "3":
            print("="*80)
            print("ANALISI BIVARIATA RISPETTO ALLA VARIABILE TARGET")
            print("="*80)
            analisi_bivariata(df)

        elif scelta == "4":
            print("="*80)
            print("ANALISI MULTIVARIATA")
            print("="*80)
            analisi_multivariata(df)

        else:
            print("Scelta non valida. Riprova.")
