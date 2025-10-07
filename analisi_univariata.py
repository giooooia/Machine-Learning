def analisi_univariata(df: Dataset):
    numeriche=df.dataset.select_dtypes(include='number').columns.tolist()
    categoriche=df.dataset.select_dtypes(include=['object','category','bool']).columns.tolist()

    #preparo lo spazio grafico

    tot = len(numeriche) + len(categoriche) #calcolo le variabili totali da rappresentare
    cols=2 #suddivido i grafici in due colonne
    rows = (tot + 1) // cols #calcolo le righe di conseguenza
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows))
    axes = axes.flatten()
    idx=0

    for col in numeriche:
        print(f"\nVariabile numerica: {col}")
        print(df.dataset[col].describe())

    for col in categoriche:
        print(f"\nVariabile categorica: {col}")
        print(df.dataset[col].value_counts())
    
    for col in categoriche:
        df.dataset[col].value_counts().plot(kind='bar', edgecolor='black', ax=axes[idx])
        axes[idx].set_title(f"Barchart di {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequenza')
        idx += 1

    for col in numeriche:
        df.dataset[col].plot(kind='hist', bins=10, edgecolor='black', ax=axes[idx])
        axes[idx].set_title(f"Istogramma di {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequenza')
        idx += 1
        
    for i in range(idx, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

