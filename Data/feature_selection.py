from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def feature_selection_by_variance(self, threshold: float = 0.1):
        """
        Elimina le feature con varianza inferiore alla soglia specificata
        e stampa quali colonne sono state rimosse.
        
        threshold: soglia minima di varianza per mantenere una feature (default 0.1)
        """
        if self.standardized:
            selector = VarianceThreshold(threshold=threshold)
            # Fit del selector
            selector.fit(self.dataset)
            
            # Colonne da mantenere e da eliminare
            support_mask = selector.get_support()
            cols_to_keep = self.dataset.columns[support_mask]
            cols_to_drop = self.dataset.columns[~support_mask]
            
            # Aggiorna il dataset
            self.dataset = self.dataset[cols_to_keep]
            
            # Stampa informazioni
            if len(cols_to_drop) > 0:
                print(f"Feature rimosse per bassa varianza ({threshold}): {list(cols_to_drop)}")
            else:
                print("Nessuna feature rimossa, tutte hanno varianza sufficiente")
            
            print(f"Dimensione dataset dopo selezione: {self.dataset.shape}")
        else:
            print("Prima è necessario standardizzare il dataset")
            
def feature_selection_kbest(self, k: int = 10, score_func = mutual_info_classif, target: str = "divorced"):
    """
    Seleziona le k feature migliori usando SelectKBest (chi2 o mutual_info_classif).
    
    k: numero di feature da mantenere
    target: nome della colonna target (default "divorced")
    """
    if target not in self.dataset.columns:
        print(f"Colonna target '{target}' non trovata nel dataset")
        return

    # Definizione X e y
    X = self.dataset.drop(columns=[target])
    y = self.dataset[target]

    # Selezione k migliori feature
    selector = SelectKBest(score_func=score_func, k=k)
    X_reduced = selector.fit_transform(X, y)

    # Identificazione delle feature selezionate
    support_mask = selector.get_support()
    cols_to_keep = X.columns[support_mask]
    cols_to_drop = X.columns[~support_mask]

    # Aggiornamento del dataset
    self.dataset = self.dataset[cols_to_keep.tolist() + [target]]

    # Output informativo
    print(f"Feature selezionate con SelectKBest (k={k}): {list(cols_to_keep)}")
    print(f"Dimensione dataset dopo selezione: {self.dataset.shape}")