from sklearn.preprocessing import StandardScaler
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
        Standardizzazione con Standard Scaler
        """
        
        if self.preprocessed:
            if not self.standardized:
                scaler = StandardScaler()
                self.dataset[:] = scaler.fit_transform(self.dataset)
                self.standardized = True
                print("Dataset standardizzato con successo")
            else:
                print("Dataset già standardizzato")
        else:
            print("Prima è necessario preprocessare il dataset")
