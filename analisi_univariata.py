"""
L’obiettivo di questa funzione (analisi_univariata) è quello di effettuare un’analisi descrittiva univariata sul dataset scelto, 
ovvero analizzare una variabile alla volta per comprendere la sua distribuzione, la presenza di eventuali valori anomali, 
la frequenza delle categorie e le principali statistiche riassuntive.
Questa fase rappresenta il primo passo dell’analisi esplorativa dei dati, utile per ottenere una panoramica generale sul 
contenuto del dataset e sulle caratteristiche delle variabili che lo compongono.

La funzione analisi_univariata(df: Dataset) riceve come input un oggetto Dataset che contiene, al suo interno, un DataFrame 
Pandas accessibile tramite l’attributo .dataset.
L’analisi è composta da tre macro-fasi:
    1) Identificazione delle variabili numeriche e categoriche
    2) Calcolo delle statistiche descrittive
    3) Visualizzazione grafica delle distribuzioni

"""

def analisi_univariata(df: Dataset):

    """prima di tutto controlliamo se il dataset è già stato preprocessato perchè se così fosse è necessario riconvertire i 
    valori 0 e 1 assunti degli attributi categorici in valori bool, in modo da poterli analizzare correttamente"""
    if df.preprocessed==True:
        for col in df.df.columns:
            valori = set(df.df[col].dropna().unique())
            if valori == {0, 1} or valori == {1, 0}:
                df.df[col] = df.df[col].replace({1: True, 0: False})
        
    numeriche=df.df.select_dtypes(include='number').columns.tolist()
    categoriche=df.df.select_dtypes(include=['object','category','bool']).columns.tolist()
    """Utilizziamo select_dtypes() per distinguere le colonne in base al loro tipo:
        - Variabili numeriche: colonne di tipo int o float;
        - Variabili categoriche: colonne di tipo object, category o bool.
        Il risultato è la creazione di due liste Python:
        - numeriche: contiene i nomi delle colonne numeriche;
        - categoriche: contiene i nomi delle colonne categoriche."""

    #preparo lo spazio grafico

    tot = len(numeriche) + len(categoriche) #calcolo le variabili totali da rappresentare
    cols=2 #suddivido i grafici in due colonne
    rows = (tot + 1) // cols #calcolo le righe di conseguenza
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows))
    axes = axes.flatten()
    idx=0

    for col in numeriche:
        print(f"\nVariabile numerica: {col}")
        print(df.df[col].describe())
    """Per ciascuna variabile numerica, viene utilizzato il metodo describe() di Pandas, che restituisce:
        - count → numero di valori non nulli;
        - mean → media aritmetica;
        - std → deviazione standard;
        - min, 25%, 50%, 75%, max → valori minimi, quartili e massimo.
      Queste misure consentono di comprendere la tendenza centrale e la dispersione dei dati."""

    for col in categoriche:
        print(f"\nVariabile categorica: {col}")
        print(df.df[col].value_counts())

    """Per ciascuna variabile categorica, viene stampata la frequenza assoluta di ogni categoria tramite value_counts().
       Questo permette di osservare la distribuzione delle modalità e di individuare eventuali squilibri o classi rare."""
    
    for col in categoriche: #Per ogni variabile categorica viene generato un grafico a barre (bar chart).
        df.df[col].value_counts().plot(kind='bar', edgecolor='black', ax=axes[idx])
        axes[idx].set_title(f"Barchart di {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequenza')
        idx += 1

    for col in numeriche: #Per ogni variabile numerica viene creato un istogramma, che suddivide i valori in 10 intervalli (bins=10).
        df.df[col].plot(kind='hist', bins=10, edgecolor='black', ax=axes[idx])
        axes[idx].set_title(f"Istogramma di {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequenza')
        idx += 1

    plt.tight_layout()
    plt.show()
    plt.savefig('analisi_univariata.png')


