import pandas as pd

def load_data():
    """Load the dataset from a CSV file."""
    data = pd.read_csv('datasets/people_wiki.csv')['text']
    print("Data loaded successfully.")
    return data