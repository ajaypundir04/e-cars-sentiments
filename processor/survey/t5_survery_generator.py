import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.log_utils import LoggerManager

class QuestionGenerator:
    def __init__(self, model_path="./t5_likert_finetuned", log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def generate_question(self, passage):
        prompt = f"Generate a Likert scale question based on the following passage: '{passage}'"
        self.logger.info(f"Generating question for passage: {passage}")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=5,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True
            )

            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print("question::::"+qu)
            print("question::::"+self.tokenizer.decode(outputs[0], skip_special_tokens=False))
            if not question.endswith("?"):
                question = f"How strongly do you agree with given statement: '{passage}'? Options: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree."

            return question

        except Exception as e:
            self.logger.error(f"Error generating question: {e}")
            return None

if __name__ == "__main__":
    question_generator = QuestionGenerator(model_path="./t5_likert_finetuned", log_level=logging.DEBUG)
    passage = "Electric vehicles are becoming more accessible to a wide range of consumers due to cheap prize"
    print(question_generator.generate_question(passage))
