import pandas as pd

def trasformazione_attributi(df):  
    """
    Trasformazione le variabili qualitative in quantitative
    """

    # Trasformo gli attributi qualitativi ordinali in discreti
    map_edu = {
            "No Formal Education": 0,
            "High School": 1,
            "Bachelor": 2,
            "Master": 3,
            "PhD": 4
        }

    df["education_level"] = df["education_level"].map(map_edu)
    
    # Trasformo gli attributi qualitativi non ordinali in codifica OneHot
    df = pd.get_dummies(df, columns=["employment_status"], dtype=int)
    df = pd.get_dummies(df, columns=["religious_compatibility"], dtype=int)
    df = pd.get_dummies(df, columns=["marriage_type"], dtype=int)
    df = pd.get_dummies(df, columns=["conflict_resolution_style"], dtype=int)
    
    print("Dataset preprocessato con successo")