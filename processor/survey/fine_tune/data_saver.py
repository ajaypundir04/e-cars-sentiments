class DataSaver:
    def __init__(self, output_path):
        self.output_path = output_path

    def save_data(self, data):
        """Saves the processed data to a CSV file."""
        data.to_csv(self.output_path, index=False)
        print(f"Data Saved to {self.output_path}")
