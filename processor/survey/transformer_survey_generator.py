import logging
from transformers import pipeline
from handler.category_handler import CategoryHandler
from langchain_ollama import OllamaLLM
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM



class TransformerSurveyGenerator:
    def __init__(self, log_level=logging.INFO):
        # Initialize logger and CategoryHandler
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.category_handler = CategoryHandler()
        #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        

        # Initialize transformer pipelines
        self.feature_extraction_pipeline = pipeline("feature-extraction", model=model)
        self.question_generation_pipeline = pipeline("text2text-generation", model=model)

    def extract_features(self, text, num_features=5):
        """Extract top features from text using a transformer model."""
        self.logger.info("Extracting features using transformers...")
        feature_scores = self.feature_extraction_pipeline(text)
        
        # Simplify extraction: Aggregate and rank features
        feature_dict = {}
        for score_set in feature_scores:
            for idx, score in enumerate(score_set):
                feature_name = f"Feature-{idx}"  # Placeholder for actual feature names
                feature_dict[feature_name] = feature_dict.get(feature_name, 0) + score
        
        # Sort features by importance and take top N
        sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)[:num_features]
        return [feature[0] for feature in sorted_features]

    def generate_questions(self, features):
        """Generate survey questions using extracted features and transformers."""
        self.logger.info("Generating survey questions using transformers...")
        questions = []
        for feature in features:
            category = self.category_handler.categorize_feature(feature)
            prompt = f"Generate a survey question about {feature} related to {category} in electric cars."
            result = self.question_generation_pipeline(prompt, max_length=64)
            questions.append(result[0]['generated_text'])
        return questions

    def generate_survey(self, text, num_features=5):
        """End-to-end survey generation process using transformers."""
        self.logger.info("Starting end-to-end survey generation...")
        
        # Extract features
        features = self.extract_features(text, num_features)
        
        # Generate survey questions
        questions = self.generate_questions(features)
        
        return {
            "features": features,
            "survey_questions": questions
        }

    def display_survey(self, survey):
        """Display survey questions with Likert scale options."""
        likert_scale = [
            "Not Important",
            "Slightly Important",
            "Moderately Important",
            "Very Important",
            "Extremely Important"
        ]
        
        print("Survey Questions:")
        for i, question in enumerate(survey['survey_questions'], start=1):
            print(f"{i}. {question}")
            print("Response options: " + ", ".join(likert_scale))
            print()

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example input text
    example_text = (
        "Electric cars are becoming increasingly popular due to their environmental benefits, "
        "including reduced emissions and improved energy efficiency. Battery technology and range "
        "are critical factors in consumer decisions, as well as affordability and charging infrastructure."
    )

    survey_generator = TransformerSurveyGenerator()
    survey = survey_generator.generate_survey(example_text, num_features=5)
    survey_generator.display_survey(survey)