from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from feature_selection import *

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

                
    def feature_selection(self):
        """
        Effettua operazioni di feature selection
        """
        feature_selection_by_variance(self)
        feature_selection_kbest(self)
        
        