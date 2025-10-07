from menu import *

def main():
    df = None
    while True:
        print("\n===== MENU =====")
        print("1. Carica dataset")
        print("2. Analisi dei dati")
        print("0. Esci")
        scelta = input("Scegli opzione: ")

        if scelta == '1':
            df = carica_dataset()
        elif scelta == '2':
            if df is not None:
                analizza_dati(df)
            else:
                print("Devi prima caricare un dataset!")
        elif scelta == '0':
            print("Uscita...")
            break
        else:
            print("Scelta non valida!")