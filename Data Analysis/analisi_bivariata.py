import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_column_types(df, target='divorced'):
    """
    Identifica le colonne numeriche e categoriche nel dataframe.
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    return num_cols, cat_cols


def plot_numeric_boxplots_grid(df, num_cols, target='divorced', cols=3):
    """
    Crea un boxplot in griglia per le variabili numeriche.

    Prende in input il dataframe, la lista degli attributi numerici, il target e il numero
    di colonne della griglia
    """
    n = len(num_cols)
    rows = (n + cols - 1) // cols  # calcola numero di righe in base a quante colonne vogliamo per riga
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    
    # Mappa temporanea per etichette
    label_map = {0: 'non divorced', 1: 'divorced'}
    
    for i, col in enumerate(num_cols):
        sns.boxplot(x=target, y=col, data=df, ax=axes[i])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        # Sostituisci etichette asse x
        axes[i].set_xticklabels([label_map[t] for t in sorted(df[target].unique())])
    
    # Rimuove subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle('Boxplot delle variabili numeriche rispetto a "divorced"', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    

def plot_categorical_bars(df, cat_cols, target='divorced'):
    """
    Crea un grafico a barre per ognuna delle variabili categoriche
    passate in input tramite lista
    """
    for col in cat_cols:
        tab = pd.crosstab(df[col], df[target], normalize='index') * 100
        tab.plot(kind='bar', stacked=True, colormap='Set2', figsize=(5,3))
        plt.title(f"{col} vs {target}")
        plt.ylabel('% divorziati / non divorziati')
        plt.xlabel(col)
        plt.legend(title=target)
        plt.tight_layout()
        plt.show()
        
        
def plot_target_correlations(df, target='divorced', annotate=True):
    """
    Visualizza le correlazioni delle feature numeriche con la variabile target.
    """
    # Calcolo delle correlazioni numeriche
    corr_target = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    corr_target = corr_target.drop(target, errors='ignore')

    # Creazione figura
    plt.figure(figsize=(8, 6))
    # Convert Series to list
    colors = corr_target.map(lambda x: 'tomato' if x > 0 else 'steelblue').tolist()

    # Grafico a barre orizzontali
    sns.barplot(
        x=corr_target.values,
        y=corr_target.index,
        palette=colors
    )

    # Titoli e label
    plt.title(f'Correlazione con la variabile target "{target}"', fontsize=14)
    plt.xlabel('Coefficiente di correlazione')
    plt.ylabel('Variabile')
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)

    # Annotazioni dei valori
    if annotate:
        for i, v in enumerate(corr_target.values):
            plt.text(
                v + 0.002 * (1 if v > 0 else -1),
                i,
                f"{v:.3f}",
                color='black',
                va='center',
                ha='left' if v > 0 else 'right',
                fontsize=9
            )

    plt.tight_layout()
    plt.show()