import logging
from handler.feature_handler import FeatureHandler
from utils.log_utils import LoggerManager
from handler.category_handler import CategoryHandler

class SurveyGenerator:
    def __init__(self, log_level=logging.INFO):
        # Initialize logger and CategoryHandler
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.feature_analysis_app = FeatureHandler(log_level)
        self.category_handler = CategoryHandler()  # New CategoryHandler instance
    
    def generate_survey(self, mode, language, keyword, num_features,file_paths = ['stats/ev_china.md','stats/ev_germany.md','stats/ev_norway.md','stats/hybrid_germany.md', 'stats/stats.md', 'stats/reviews.csv']):
        """
        Generates survey questions based on the top features extracted from FeatureHandler.
        
        Args:
            mode (str): The mode for feature extraction ('url', 'file', or 'all').
            language (str): Language code for analysis, e.g., 'EN', 'DE'.
            keyword (str): Keyword to search for in the analysis.
            num_features (int): Number of top features to extract and base the survey on.
            file_paths (list): List of file paths to process in 'file' mode. Default is ['stats/ev_china.md'].
        
        Returns:
            list: A list of improved survey questions.
        """
        # List to hold all top features from both file and URL modes
        top_features = []

        if mode == 'file' or mode == 'all':
            # Process the file mode
            top_features.append(self.feature_analysis_app.process_from_file(keyword, language, num_features, file_paths))
        
        if mode == 'url' or mode == 'all':
            # Process the URL mode
            top_features.append(self.feature_analysis_app.process_from_url(language, keyword, num_features))

        # Flatten the list if top_features is nested, and only keep the feature names (not counts)
        flat_features = [feature[0] if isinstance(feature, tuple) else feature for sublist in top_features for feature in sublist]

        # Remove duplicates
        flat_features = list(set(flat_features))

        # Generate improved survey questions based on features
        survey_questions = []
        for feature in flat_features:
            question = self.create_question_from_feature(feature)
            survey_questions.append(question)
        
        return {
            "survey_questions":survey_questions,
            "features":top_features
        }

    def create_question_from_feature(self, feature):
        """
        Creates a survey question based on the given feature and its category.
        
        Args:
            feature (str): A feature extracted from the data.
        
        Returns:
            str: A formatted survey question.
        """
        # Categorize the feature using CategoryHandler
        category = self.category_handler.categorize_feature(feature)

        # Generate different types of questions based on the category
        if category == 'performance':
            templates = [
                f"How satisfied are you with the {feature} performance of your electric car?",
                f"Do you believe the {feature} is a key factor in electric car performance?",
                f"How important is the {feature} when evaluating your electric car’s overall efficiency?"
            ]
        elif category == 'design':
            templates = [
                f"How would you rate the {feature} design of your electric car?",
                f"How important is the {feature} design in your decision to purchase this electric car?",
                f"Does the {feature} design enhance the electric car’s value for you?"
            ]
        elif category == 'usability':
            templates = [
                f"How easy is it to use the electric car with its {feature}?",
                f"How important is {feature} for improving the usability of the electric car?",
                f"Do you find the {feature} makes the electric car more user-friendly?"
            ]
        elif category == 'affordability':
            templates = [
                f"How would you rate the {feature} in terms of electric car affordability?",
                f"Do you think the {feature} offers good value for its price in an electric car?",
                f"How important is {feature} when considering the overall cost of your electric car?"
            ]
        # Add more categories based on CategoryHandler
        elif category == 'safety':
            templates = [
                f"How would you rate the safety features related to {feature} in your electric car?",
                f"Do you feel the {feature} adds to the overall safety of your electric car?",
                f"How important is the {feature} for your feeling of safety in the electric car?"
            ]
        else:  # General questions
            templates = [
                f"How important is the '{feature}' when considering purchasing an electric car?",
                f"How likely are you to choose an electric car that offers better '{feature}'?",
                f"To what extent does '{feature}' influence your decision when comparing electric cars?"
            ]
        
        # Choose a template based on the feature (for example, rotate templates)
        return templates[hash(feature) % len(templates)]
    
    def display_survey(self, survey_questions):
        """
        Display survey questions with Likert scale options.

        Args:
            survey_questions (list): A list of survey questions to display.
        
        Returns:
            None
        """
        likert_scale = ["Not Important", "Slightly Important", "Moderately Important", "Very Important", "Extremely Important"]
        
        print("Survey Questions:")
        for i, question in enumerate(survey_questions, start=1):
            print(f"{i}. {question}")
            print("Response options: " + ", ".join(likert_scale))
            print("\n")
