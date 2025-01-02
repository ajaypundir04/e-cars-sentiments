class DataProcessor:
    def __init__(self, data, question_generator):
        self.data = data
        self.question_generator = question_generator

    def process_data(self, context_column):
        """Generates questions for each row in the dataset."""
        questions = []
        for context in self.data[context_column]:
            question = self.question_generator.generate_question(context)
            questions.append(question)
        self.data['generated_questions'] = questions
        print("Questions Generated Successfully")
        return self.data
