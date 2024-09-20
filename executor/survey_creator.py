import argparse
from processor.survey.survey_generator import SurveyGenerator  # Assuming this is the correct import for the SurveyGenerator class
from output.survey_printer import SurveyPrinter

class SurveyApp:
    def __init__(self):
        """
        Initialize the SurveyApp with default values and argument parser.
        """
        self.survey_generator = SurveyGenerator()  # Initialize the SurveyGenerator

    def parse_arguments(self):
        """
        Parse command-line arguments using argparse and return them.
        """
        parser = argparse.ArgumentParser(description="Generate a survey based on extracted features.")
        
        # Define arguments
        parser.add_argument('--mode', type=str, choices=['url', 'file'], default='file',
                            help="Mode to run the analysis: 'url' or 'file'. Default is 'file'.")
        parser.add_argument('--language', type=str, required=True,
                            help="Language code to use for analysis, e.g., 'EN', 'DE'.")
        parser.add_argument('--keyword', type=str, default='cars',
                            help="Keyword to search for in the analysis. Default is 'cars'.")
        parser.add_argument('--num_features', type=int, default=5,
                            help="Number of top features to extract and base the survey on. Default is 5.")
        
        # Parse arguments
        return parser.parse_args()

    def run(self):
        """
        Run the survey generation process by parsing arguments and generating the survey.
        """
        # Parse the arguments
        args = self.parse_arguments()

        # Generate the survey questions using the arguments
        survey = self.survey_generator.generate_survey(
            mode=args.mode,
            language=args.language,
            keyword=args.keyword,
            num_features=args.num_features
        )

       # Check if the survey is a dictionary and extract data accordingly
        survey_questions = survey.get('survey_questions')
        features_with_occurrences = survey.get('features')

        # Display the generated survey questions
        self.survey_generator.display_survey(survey_questions)

        # Plot the features and their corresponding questions
        SurveyPrinter.plot_feature_frequencies(features_with_occurrences)
        SurveyPrinter.plot_survey_questions(survey_questions)



# Example usage when running the script directly
if __name__ == "__main__":
    app = SurveyApp()  # Initialize the SurveyApp
    app.run()  # Run the survey generation process
