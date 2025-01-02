from abc import ABC, abstractmethod

class AbstractSurveyGenerator(ABC):
    """
    Abstract base class for generating surveys based on extracted features.
    """
    
    @abstractmethod
    def generate_survey(self, mode, language, keyword, num_features, file_paths):
        """
        Abstract method to generate the survey questions based on the features.
        
        Args:
            mode (str): Mode to run the analysis ('url' or 'file').
            language (str): Language code for the analysis.
            keyword (str): Keyword for the analysis.
            num_features (int): Number of top features to extract.
            file_paths (list): List of file paths (if mode is 'file').
        
        Returns:
            dict: A dictionary containing 'survey_questions' and 'features' (top features).
        """
        pass

    @abstractmethod
    def display_survey(self, survey_questions):
        """
        Abstract method to display the survey questions.
        
        Args:
            survey_questions (list): The survey questions to display.
        """
        pass
