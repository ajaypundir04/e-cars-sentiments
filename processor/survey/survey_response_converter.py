import configparser
import pandas as pd

class SurveyResponseConverter:
    def __init__(self, csv_file_path, ini_file_path, lang='EN'):
        # Define the reverse mapping from numerical values to textual responses
        self.response_mapping = {
            1: 'Strongly Disagree',
            2: 'Disagree',
            3: 'Neutral',
            4: 'Agree',
            5: 'Strongly Agree'
        }
        
        # Read the CSV file
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(self.csv_file_path)

        # Strip any extra spaces from column names
        self.df.columns = self.df.columns.str.strip()

        # Define the questions corresponding to the columns (except the Timestamp column)
        self.questions = [
            'How likely is electric car adoption to be accelerated by government policies?',
            'How likely is the expansion of EV charging networks to be?',
            'How likely are advances in battery technology to increase the driving range of electric cars?',
            'How likely are automakers to commit to full electrification by 2035?',
            'How likely is electric car preference to be related to climate change?',
            'How likely is it that electric cars will be able to be used in the future?',
            'Is it likely that electric cars will be more popular in rural areas?',
            'How likely are some consumers to be discouraged from switching to electric cars?',
            'How likely is it that charging an electric car will cause more inconvenience than refueling gasoline?',
            'How likely is it that potential buyers will be reluctant to invest in an electric car?'
        ]
        
        # Initialize the INI file path
        self.ini_file_path = ini_file_path
        self.lang=lang

    def convert_responses(self):
        """
        Convert the numerical responses from the CSV file back into textual responses.
        """
        survey_responses = []
        
        # Iterate through the rows in the DataFrame (excluding the Timestamp column)
        for _, row in self.df.iterrows():
            response_data = {}
            for idx, question in enumerate(self.questions):
                # Check if the column exists in the DataFrame
                if question in self.df.columns:
                    response_data[f'question{idx + 1}'] = self.response_mapping.get(row[question], None)
                else:
                    # If the question column is missing, handle the error gracefully
                    response_data[f'question{idx + 1}'] = None  # Or you can use a default value like "Unknown"
            survey_responses.append(response_data)
        
        return survey_responses
    
    def update_ini_file(self):
        """
        Read the existing INI file, replace the survey_responses section, and save the updated file.
        """
        survey_responses = self.convert_responses()

        # Prepare the string representation for the survey_responses in the exact format
        survey_responses_str =  str(survey_responses).replace("},", "},\n\t").replace("{", "{ ").replace("}", " }")

        # Initialize config parser
        config = configparser.ConfigParser()

        # Read the existing INI file (if it exists)
        try:
            config.read(self.ini_file_path)
        except FileNotFoundError:
            print(f"Warning: {self.ini_file_path} not found. A new file will be created.")
        
       
        
        # Add the necessary keys and values to the INI file
       
        config.set(self.lang, 'survey_responses', survey_responses_str)


        # Write the updated config to the INI file
        with open(self.ini_file_path, 'w') as configfile:
            config.write(configfile)
        print(f"Survey responses have been updated in {self.ini_file_path}")

# Example usage
if __name__ == "__main__":
    # Define the path to the input CSV file and the existing INI file
    csv_file_path = 'survey_response.csv'
    ini_file_path = 'survey.ini'
    
    # Create an instance of the SurveyResponseConverter class
    converter = SurveyResponseConverter(csv_file_path, ini_file_path)
    
    # Update the survey responses in the existing INI file
    converter.update_ini_file()
