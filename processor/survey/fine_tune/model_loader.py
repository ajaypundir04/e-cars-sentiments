from transformers import T5ForConditionalGeneration, T5Tokenizer

class ModelLoader:
    def __init__(self, model_name="t5-small"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Loads the T5 model and tokenizer."""
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        print(f"Model '{self.model_name}' Loaded Successfully")
        return self.model, self.tokenizer
