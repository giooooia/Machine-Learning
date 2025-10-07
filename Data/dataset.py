from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

class Dataset:
    def __init__(self, cvs_path: str):
        self.dataset = pd.read_csv(cvs_path)
        self.preprocessed = False
        self.standardized = False

    def preProcessing(self):  
        """
        Trasformazione le variabili qualitative in quantitative
        """
        
        if not self.preprocessed:
            map_edu = {
                "No Formal Education": 0,
                "High School": 1,
                "Bachelor": 2,
                "Master": 3,
                "PhD": 4
            }

            # Trasformo gli attributi qualitativi ordinali in discreti
            self.dataset["education_level"] = self.dataset["education_level"].map(map_edu)
            self.dataset = pd.get_dummies(self.dataset, columns=["employment_status"], dtype=int)
            self.dataset = pd.get_dummies(self.dataset, columns=["religious_compatibility"], dtype=int)
            self.dataset = pd.get_dummies(self.dataset, columns=["marriage_type"], dtype=int)
            self.dataset = pd.get_dummies(self.dataset, columns=["conflict_resolution_style"], dtype=int)

            self.preprocessed = True
            print("Dataset preprocessato con successo")
        else:
            print("Preprocessing già effettuato")

    def standardize(self):
        """
        Standardizzazione solo delle colonne numeriche continue.
        """
        if self.preprocessed:
            if not self.standardized:
                # Seleziona solo colonne numeriche originali
                numeric_cols = ["age_at_marriage", "marriage_duration_years", "num_children",
                                "education_level", "combined_income", "cultural_background_match",
                                "communication_score", "conflict_frequency", "financial_stress_level",
                                "mental_health_issues"]  # adatta alla tua lista
                scaler = StandardScaler()
                self.dataset[numeric_cols] = scaler.fit_transform(self.dataset[numeric_cols])
                self.standardized = True
                print("Dataset standardizzato con successo")
            else:
                print("Dataset già standardizzato")
        else:
            print("Prima è necessario preprocessare il dataset")

                
    def feature_selection_by_variance(self, threshold: float = 0.1):
        """
        Elimina le feature con varianza inferiore alla soglia specificata
        e stampa quali colonne sono state rimosse.
        
        :param threshold: soglia minima di varianza per mantenere una feature (default 0.1)
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
