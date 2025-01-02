from transformers import pipeline
import logging
from utils.log_utils import LoggerManager  

class LikertQuestionGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", log_level=logging.INFO):
        """
        Initializes the LikertQuestionGenerator class with a specified model and logging.

        Args:
            model_name (str): Name of the text generation model to use (default: "gpt-3.5-turbo").
            log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        # Initialize logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        
        # Load the text generation pipeline
        self.logger.info("Initializing text generation model.")
        self.generator = pipeline("text-generation", model=model_name)

        

    def generate_likert_questions(self, features, custom_prompt=None):
        """
        Generates Likert scale questions based on provided features related to electric cars.
        Optionally accepts a custom prompt to refine question generation.

        Args:
            features (list): A list of features (strings) for which Likert scale questions will be generated.
            custom_prompt (str): A custom prompt to influence the question generation.

        Returns:
            list: A list of Likert scale questions generated for each feature.
        """
        

        # Expanded set of templates focused on electric car features
        templates = [
            "How satisfied are you with the {feature} of the electric car?",
            "On a scale from 1 to 5, how would you rate the {feature} of this electric vehicle?",
            "How important do you consider the {feature} when evaluating this electric car?",
            "To what extent does the {feature} meet your expectations in an electric vehicle?",
            "How likely are you to recommend an electric car based on its {feature}?",
            "How well does the {feature} contribute to your overall satisfaction with the electric vehicle?",
            "How would you rate the quality of the {feature} in this electric vehicle?",
            "To what degree does the {feature} enhance your driving experience with the electric car?"
        ]

        # Store the questions
        questions = []

        for feature in features:
            # Normalize the feature for better grammatical coherence
            feature_normalized = self._normalize_feature(feature)

            # Select a template based on the feature (ensure some diversity)
            template = templates[hash(feature_normalized) % len(templates)]
            prompt = template.format(feature=feature_normalized)

            if custom_prompt:
                # Use custom prompt if provided
                prompt = f"{custom_prompt} {feature_normalized}"

            try:
                self.logger.info(f"Generating question for feature: {feature_normalized}")
                question = self.generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']

                # Clean up the generated question to ensure it makes sense
                question = self._clean_generated_question(question)
                questions.append(question)
            except Exception as e:
                self.logger.error(f"Error generating question for feature '{feature}': {e}")
                questions.append(f"Error generating question for {feature}")
        
        return questions

  

    def _normalize_feature(self, feature):
        """
        Normalizes the feature string for grammatical consistency.
        This method handles pluralization and general string formatting.

        Args:
            feature (str): The feature to be normalized.

        Returns:
            str: The normalized feature.
        """
        # Capitalize first letter for consistency
        feature = feature.strip().capitalize()

        # Add more normalization logic here if needed, such as handling plurals, etc.
        return feature

    def _clean_generated_question(self, question):
        """
        Cleans up the generated question text to ensure it is coherent and natural.
        
        Args:
            question (str): The generated question text.
        
        Returns:
            str: The cleaned-up question.
        """
        # Trim any leading or trailing spaces or incomplete sentences
        question = question.strip()
        
        # Ensure question ends with a question mark
        if not question.endswith("?"):
            question += "?"
        
        return question
