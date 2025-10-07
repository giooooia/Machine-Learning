import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analisi_bivariata(df, target='divorced'):
    """
    Esegue un'analisi bivariata completa tra le variabili indipendenti e la variabile target.
    Mostra boxplot, distribuzioni categoriche e correlazioni con il target.
    """
    print("="*80)
    print(f"ANALISI BIVARIATA RISPETTO ALLA VARIABILE TARGET: '{target.upper()}'")
    print("="*80)

    print("\n[1] Identificazione delle colonne numeriche e categoriche...\n")
    num_cols, cat_cols = get_column_types(df, target)
    print(f"Variabili numeriche trovate ({len(num_cols)}): {num_cols}")
    print(f"Variabili categoriche trovate ({len(cat_cols)}): {cat_cols}")

    # Analisi variabili numeriche
    if num_cols:
        print("\n" + "-"*80)
        print("ANALISI VARIABILI NUMERICHE (Boxplot rispetto al target)\n")
        print("Ogni grafico mostra la distribuzione di una variabile numerica\n"
              f"tra le due classi del target '{target}'.\n")
        plot_numeric_boxplots_grid(df, num_cols, target)
    else:
        print("\nNessuna variabile numerica trovata da analizzare.\n")

    # Analisi variabili categoriche
    if cat_cols:
        print("\n" + "-"*80)
        print("ANALISI VARIABILI CATEGORICHE (Distribuzione percentuale rispetto al target)\n")
        print("Ogni grafico mostra la distribuzione percentuale delle categorie\n"
              f"per le due classi del target '{target}'.\n")
        plot_target_distribution_by_categories(df, cat_cols, target)
    else:
        print("\n Nessuna variabile categorica trovata da analizzare.\n")

    # Correlazioni con il target
    print("\n" + "-"*80)
    print("CORRELAZIONI DELLE VARIABILI NUMERICHE CON IL TARGET\n")
    print("Mostra la forza e la direzione (positiva/negativa) della correlazione lineare\n"
          f"tra le variabili numeriche e la variabile target '{target}'.\n")
    plot_target_correlations(df, target)

    print("\n" + "="*80)
    print("Analisi bivariata completata con successo\n")


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
    Sostituisce 0 e 1 con 'non divorced' e 'divorced' sull'asse x senza warnings.
    """
    n = len(num_cols)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    
    # Trasforma il target in categorico con nomi leggibili
    df_plot = df.copy()
    df_plot[target] = df_plot[target].map({0: 'non divorced', 1: 'divorced'}).astype('category')
    
    for i, col in enumerate(num_cols):
        sns.boxplot(x=target, y=col, data=df_plot, ax=axes[i])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    
    # Rimuove subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle('Boxplot delle variabili numeriche rispetto a "divorced"', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    

def plot_target_distribution_by_categories(df, cat_cols, target='divorced'):
    """
    Per ogni variabile categorica nella lista cat_cols, mostra la distribuzione dei valori
    all'interno di ciascuna classe del target come barre impilate orizzontali con percentuali.
    Sull'asse x compaiono solo le etichette 'Divorziato' e 'Non divorziato'.
    """
    for cat_col in cat_cols:
        # Tabella percentuale normalizzata per colonna (target)
        tab = pd.crosstab(df[cat_col], df[target], normalize='columns') * 100

        # Impostazioni grafico
        fig, ax = plt.subplots(figsize=(8, 5))

        # Barre impilate (orizzontali)
        bottom = np.zeros(tab.shape[1])
        colors = plt.get_cmap("Set2").colors  # palette di colori

        for i, cat_value in enumerate(tab.index):
            ax.bar(tab.columns, tab.loc[cat_value], bottom=bottom, label=cat_value, color=colors[i % len(colors)])

            # Annotazioni percentuali
            for j, val in enumerate(tab.loc[cat_value]):
                if val > 0:
                    ax.text(
                        j,
                        bottom[j] + val / 2,
                        f"{val:.1f}%",
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='black'
                    )
            bottom += tab.loc[cat_value].values

        # Titoli e label
        ax.set_ylabel('% all’interno del target', fontsize=12)
        ax.set_xlabel('Stato civile', fontsize=12)
        ax.set_title(f'Distribuzione di "{cat_col}" per {target}', fontsize=14)

        # Etichette asse x personalizzate
        x_labels = ['Non divorziato', 'Divorziato']
        ax.set_xticks(range(len(tab.columns)))
        ax.set_xticklabels(x_labels, fontsize=11)

        # Legenda fuori dal grafico
        ax.legend(title=cat_col, bbox_to_anchor=(1.05, 1), loc='upper left')

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
        palette=colors,
        hue=corr_target.index,  # aggiunto per evitare warning
        dodge=False,            # evita la separazione in barre multiple
        legend=False
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
