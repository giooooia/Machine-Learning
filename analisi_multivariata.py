import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from Dataset import Dataset importare il file che contiene la classe dataset

def analisi_multivariata(df: Dataset, fig_size: list = [8, 6]):

    df.dataset = df.dataset.drop(columns=['divorced'], errors='ignore') # per l'analisi droppiamo le labels
    # correlazione lineare
    corr = df.dataset.corr()
    plt.figure(figsize=fig_size)
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title('Matrice di correlazione (Pearson)')
    plt.show()

    # correlazione non lineare
    plt.figure(figsize=fig_size)
    corr_spearman = df.dataset.corr(method='spearman')
    sns.heatmap(corr_spearman,cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title('Correlazione di Spearman (non lineare)')
    plt.show()