import pandas as pd

class Dataset:
    def __init__(self, cvs_path: str):
        self.dataset = pd.read_csv(cvs_path)
        self.preporcessed = False
        self.standardized = False

    def preProcessing(self):
        if self.preporcessed == False:
            map_edu = {
                "No Formal Education": 0,
                "High School": 1,
                "Bachelor": 2,
                "Master": 3,
                "PhD": 4
                }

            # tarasformo gli attributi qualitativi ordinali in discreti
            self.dataset["education_level"] = self.dataset["education_level"].map(map_edu)
            self.dataset = pd.get_dummies(self.dataset, columns=["employment_status"])
            self.dataset = pd.get_dummies(self.dataset, columns=["religious_compatibility"])
            self.dataset = pd.get_dummies(self.dataset, columns=["marriage_type"])
            self.dataset = pd.get_dummies(self.dataset, columns=["conflict_resolution_style"])

            self.preporcessed = True
        else:
            print("Preprocessing già effettuato")