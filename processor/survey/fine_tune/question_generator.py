class QuestionGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_question(self, context, max_length=50):
        """Generates a question based on the provided context."""
        input_text = f"generate question: {context}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question
