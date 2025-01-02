import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads data from a CSV file."""
        self.data = pd.read_csv(self.file_path)
        print("Data Loaded Successfully")
        return self.data
