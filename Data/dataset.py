from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
                               from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
import pandas as pd

class Dataset:
    def __init__(self, csv_path: str):
        self.dataset = pd.read_csv(csv_path)
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
            
    def split(self):
        """"
        Splitta il dataset in training set(80%) e test set(20%)
        """
        
        if self.preprocessed:
            X = self.dataset.drop(columns=['divorced'])
            y = self.dataset['divorced']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30,
                                                                stratify= y)
            return X_train, X_test, y_train, y_test
            
        else: print("Operazione non riuscita. E' necessario preprocessare prima il dataset")
        
        
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
        print(f"Dimensione dataset dopo selezione: {self.dataset.shape}") "communication_score", "conflict_frequency", "financial_stress_level",
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