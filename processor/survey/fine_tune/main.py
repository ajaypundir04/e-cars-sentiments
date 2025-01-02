from .data_loader import DataLoader
from .model_loader import ModelLoader
from .question_generator import QuestionGenerator
from .data_processor import DataProcessor
from .data_saver import DataSaver


def main():
    # Step 1: Load Data
    data_loader = DataLoader("processor/survey/fine_tune/reviews.csv")
    data = data_loader.load_data()

    # Step 2: Load Model and Tokenizer
    model_loader = ModelLoader(model_name="t5-small")
    model, tokenizer = model_loader.load_model()

    # Step 3: Initialize Question Generator
    question_generator = QuestionGenerator(model, tokenizer)

    # Step 4: Process Data and Generate Questions
    data_processor = DataProcessor(data, question_generator)
    processed_data = data_processor.process_data(context_column="Review_Text")

    # Step 5: Save the Output
    data_saver = DataSaver("dataset_with_questions.csv")
    data_saver.save_data(processed_data)

if __name__ == "__main__":
    main()
