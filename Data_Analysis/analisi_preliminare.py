import pandas as pd
import numpy as np

from Data.dataset import Dataset


def statistiche_generali(df: Dataset):

    shp = df.dataset.shape
    print(f'Il dataset ha {shp[0]} record da {shp[1]} attributi ciascuno\n\n')

    print('Di seguito il numero di valori nulli per attributo nel dataset:', df.dataset.isnull().sum(), sep='\n', end='\n\n')

    print('Di seguito il il tipo di dato di ciascun attributo nel dataset:', df.dataset.dtypes, sep='\n', end='\n\n')
    
    num_attributi = df.dataset['divorced'].value_counts()
    print(f'Il dataset ha {num_attributi[0]} attributi per la classe 0 e {num_attributi[1]} attributi per la classe 1')
    
    print("\n" + "="*80)
    print("Analisi preliminare completata con successo\n")

