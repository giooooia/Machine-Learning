import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

df = pd.read_csv("/home/giulia/Desktop/ML/progettoML/Machine-Learning/Data/divorce_df.csv", encoding='UTF-8')

# Analisi descrittiva dei singoli attributi
numeriche = df.select_dtypes(include='number').columns.tolist()
categoriche = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

def plot_univariata(df, numeriche, categoriche, filename="analisi_univariata.png"):
    tot = len(numeriche) + len(categoriche)
    cols = 2
    rows = (tot + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows))
    axes = axes.flatten()

    idx = 0
    for col in numeriche:
        df[col].plot(kind='hist', bins=10, edgecolor='black', ax=axes[idx])
        axes[idx].set_title(f"Istogramma di {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequenza')
        idx += 1

    for col in categoriche:
        df[col].value_counts().plot(kind='bar', edgecolor='black', ax=axes[idx])
        axes[idx].set_title(f"Barplot di {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequenza')
        idx += 1

    # Rimuovi assi vuoti
    for i in range(idx, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Stampa statistiche descrittive
for col in numeriche:
    print(f"\nVariabile numerica: {col}")
    print(df[col].describe())

for col in categoriche:
    print(f"\nVariabile categorica: {col}")
    print(df[col].value_counts())

# Crea unico file con tutti i grafici
plot_univariata(df, numeriche, categoriche)

