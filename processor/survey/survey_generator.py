import logging
import random
from handler.feature_handler import FeatureHandler
from utils.log_utils import LoggerManager
from handler.category_handler import CategoryHandler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from processor.survey.likert_test import LikertQuestionModel  # Import the LikertQuestionModel

class SurveyGenerator:
    def __init__(self, log_level=logging.INFO):
        # Initialize logger and CategoryHandler
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.feature_analysis_app = FeatureHandler(log_level)
        self.category_handler = CategoryHandler()  # New CategoryHandler instance
        
        # Initialize the LikertQuestionModel instance
        self.model = LikertQuestionModel(train_file="train.jsonl", 
                                         valid_file="valid.jsonl", 
                                         output_dir="./t5_likert_finetuned_ev")
        
        # Tokenizer and Model for handling feature-to-question generation
        self.tokenizer = T5Tokenizer.from_pretrained('allenai/t5-small-squad2-question-generation')
        self.model_name = 'allenai/t5-small-squad2-question-generation'
        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def generate_survey(self, mode, language, keyword, num_features, file_paths=None):
        if file_paths is None:
            file_paths = ['stats/ev_china.md', 'stats/ev_germany.md', 'stats/ev_norway.md', 
                          'stats/hybrid_germany.md', 'stats/stats.md', 'stats/reviews.csv']
        
        top_features = []
        if mode in ['file', 'all']:
            top_features.append(self.feature_analysis_app.process_from_file(keyword, language, num_features, file_paths))
        if mode in ['url', 'all']:
            top_features.append(self.feature_analysis_app.process_from_url(language, keyword, num_features))
        
        # Flatten and deduplicate feature list
        flat_features = list(set(feature[0] if isinstance(feature, tuple) else feature for sublist in top_features for feature in sublist))
        
        # Generate survey questions using LikertQuestionModel
        survey_questions = []
        for feature in flat_features:
            question = self.create_question_from_feature(feature)
            survey_questions.append(question)
        
        return {"survey_questions": survey_questions, "features": top_features}

    def create_question_from_feature(self, feature):
        category = self.category_handler.categorize_feature(feature)
        templates = self._get_template_for_category(category, feature)
        
        # Use LikertQuestionModel to generate the question
        generated_question = self._generate_dynamic_question(feature, category, templates)
        return generated_question

    def _get_template_for_category(self, category, feature):
        template_dict = {
            "performance": [
                f"How satisfied are you with the {feature} performance of your electric car?",
                f"Do you believe the {feature} is a key factor in electric car performance?",
                f"How important is the {feature} when evaluating your electric car’s overall efficiency?",
                f"Would you recommend an electric car based on its {feature} performance?",
                f"How would you rate the {feature} performance compared to similar models?",
            ],
            "design": [
                f"How would you rate the {feature} design of your electric car?",
                f"How important is the {feature} design in your decision to purchase this electric car?",
                f"Does the {feature} design enhance the electric car’s value for you?",
                f"Do you find the {feature} design appealing and modern?",
                f"How much does the {feature} design affect your satisfaction with the vehicle?",
            ],
            "usability": [
                f"How easy is it to use the electric car with its {feature}?",
                f"How important is {feature} for improving the usability of the electric car?",
                f"Do you find the {feature} makes the electric car more user-friendly?",
                f"How intuitive is the {feature} in your day-to-day use of the electric car?",
                f"How much does the {feature} contribute to the overall convenience of the vehicle?",
            ],
            "affordability": [
                f"How would you rate the {feature} in terms of electric car affordability?",
                f"Do you think the {feature} offers good value for its price in an electric car?",
                f"How important is {feature} when considering the overall cost of your electric car?",
                f"Does the {feature} pricing align with your expectations for affordability?",
                f"Would you consider the {feature} as a deciding factor in your purchase decision?",
            ],
            "safety": [
                f"How would you rate the safety features related to {feature} in your electric car?",
                f"Do you feel the {feature} adds to the overall safety of your electric car?",
                f"How important is the {feature} for your feeling of safety in the electric car?",
                f"How confident are you in the {feature} for ensuring passenger safety?",
                f"Does the {feature} meet your expectations for safety standards in electric vehicles?",
            ],
        }
        return template_dict.get(category, [
            f"How important is the '{feature}' when considering purchasing an electric car?",
            f"How likely are you to choose an electric car that offers better '{feature}'?",
            f"To what extent does '{feature}' influence your decision when comparing electric cars?",
            f"Would the {feature} of an electric car significantly affect your purchase decision?",
            f"Do you believe that '{feature}' differentiates electric cars from other options?",
        ])
    
    def _generate_dynamic_question(self, feature, category, templates):
        # Generate a Likert scale question using LikertQuestionModel
        generated_question = self.model.generate_question(feature)
        if len(generated_question) < 10:  # Check if the question is too short
            self.logger.warning(f"Generated question for feature '{feature}' is too short. Using fallback template.")
            generated_question = random.choice(templates)  # Use a fallback template if the question is too short
        return generated_question

    def display_survey(self, survey_questions):
        likert_scale = ["Not Important", "Slightly Important", "Moderately Important", 
                        "Very Important", "Extremely Important"]
        print("Survey Questions:")
        for i, question in enumerate(survey_questions, start=1):
            print(f"{i}. {question}")
            print("Response options: " + ", ".join(likert_scale))
            print("\n")
