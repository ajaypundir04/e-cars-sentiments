import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
from utils.log_utils import LoggerManager


trained_model_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'
trained_tokenizer_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'


class QuestionGeneration:

    def __init__(self, model_dir=None, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_contexts_answers(self, json_file):
        try:
            with open(json_file, 'r') as f:
                contexts_answers = json.load(f)
            return contexts_answers
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {e}")
            return {}

    def generate(self, answer: str, context: str):
        input_text = '<answer> %s <context> %s ' % (answer, context)
        encoding = self.tokenizer.encode_plus(
            input_text,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        question = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return {'question': question, 'answer': answer, 'context': context}

    def generate_questions(self, contexts_answers):
        generated_questions = []

        # Iterate over the dictionary of context-answer pairs
        for key, value in contexts_answers.items():
            context = value["context"]
            answer = value["answer"]

            # Generate the question
            qa = self.generate(answer, context)
            question = qa['question']

            # Append the generated question to the list
            generated_questions.append(question)

        return generated_questions


if __name__ == "__main__":
    # Initialize QuestionGeneration instance with the path to the JSON file
    questionGeneration = QuestionGeneration()

    # Get the list of generated questions
    questions = questionGeneration.generate_questions(questionGeneration.load_contexts_answers(
        'contexts_answers.json'
    ))

    # Print the generated questions
    for q in questions:
        print(q['question'])
