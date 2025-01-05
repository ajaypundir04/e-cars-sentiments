import logging
from utils.log_utils import LoggerManager


class QuestionGenerator:
    def __init__(self, model, tokenizer, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer

    def generate_question(self, passage, question_template="How likely this is useful for electric cars?"):
        self.logger.info("Generating question from passage...")
        input_text = f"question: {question_template} context: {passage}"
        encoded_input = self.tokenizer([input_text], return_tensors='pt', max_length=512, truncation=True)

        output = self.model.generate(
            input_ids=encoded_input.input_ids,
            attention_mask=encoded_input.attention_mask,
            max_length=128,
            num_beams=5,
            do_sample=False,
            early_stopping=True
        )

        generated_question = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        self.logger.info("Question generation complete.")
        return generated_question
